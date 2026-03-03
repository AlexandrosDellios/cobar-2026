import numpy as np
from scipy.interpolate import CubicSpline
from flygym import assets_dir


class PreprogrammedSteps:
    """Preprogrammed steps by each leg extracted from experimental
    recordings.

    Attributes
    ----------
    legs : list[str]
        List of leg names (e.g. LF for left front leg).
    dofs_per_leg : list[str]
        List of names for degrees of freedom for each leg.
    duration : float
        Duration of the preprogrammed step (at 1x speed) in seconds.
    neutral_pos : dict[str, np.ndarray]
        Neutral position of DoFs for each leg. Keys are leg names; values
        are joint angles in the order of ``self.dofs_per_leg``.
    swing_period : dict[str, np.ndarray]
        The start and end of the lifted swing phase for each leg. Keys are
        leg names; values are arrays of shape (2,) with the start and end
        of the swing normalized to [0, 2π].

    Parameters
    ----------
    path : str or Path, optional
        Path to the preprogrammed steps data. If None, the default
        preprogrammed steps data will be loaded.
    neutral_pose_phases : list[float]
        Phase during the preprogrammed step that should be considered the
        "neutral" resting pose. This is specified for each of the 6 limbs
        and normalized to [0, 2π).
    """

    legs = [f"{side}{pos}" for side in "lr" for pos in "fmh"]

    def __init__(
        self, path=None, neutral_pose_phases=(np.pi, np.pi, np.pi, np.pi, np.pi, np.pi)
    ):
        single_step_data = np.load(
            assets_dir / "demo" / "single_steps_untethered.npz", allow_pickle=True
        )
        self._length = len(single_step_data["dof_angles"])
        self._timestep = single_step_data["timestep"]
        self.duration = self._length * self._timestep

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for i, leg in enumerate(self.legs):
            joint_angles = single_step_data["dof_angles"][:, i * 7 : (i + 1) * 7].T
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, joint_angles, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta_neutral)[:, np.newaxis]
            for leg, theta_neutral in zip(self.legs, neutral_pose_phases)
        }

        stance_time_dict = single_step_data["stance_time"].item()
        self.swing_period = {}
        for leg in self.legs:
            my_swing_period = np.array([0, stance_time_dict[leg]])
            my_swing_period /= self.duration
            my_swing_period *= 2 * np.pi
            self.swing_period[leg] = my_swing_period
        self._swing_start_arr = np.array(
            [self.swing_period[leg][0] for leg in self.legs]
        )
        self._swing_end_arr = np.array([self.swing_period[leg][1] for leg in self.legs])

    def get_joint_angles(self, leg, phase, magnitude=1):
        """Get joint angles for a given leg at a given phase.

        Parameters
        ----------
        leg : str
            Leg name.
        phase : float or np.ndarray
            Phase or array of phases of the step normalized to [0, 2π].
        magnitude : float or np.ndarray, optional
            Magnitude of the step. Default: 1 (the preprogrammed steps as
            provided).

        Returns
        -------
        np.ndarray
            Joint angles of the leg at the given phase(s). The shape of the
            array is (7, n) if ``phase`` is a 1D array of n elements, or
            (7,) if ``phase`` is a scalar.
        """
        if isinstance(phase, float) or isinstance(phase, int) or phase.shape == ():
            phase = np.array([phase])
        psi_func = self._psi_funcs[leg]
        offset = psi_func(phase) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        return joint_angles.squeeze()

    def get_adhesion_onoff(self, leg, phase):
        """Get whether adhesion is on for a given leg at a given phase.

        Parameters
        ----------
        leg : str
            Leg name.
        phase : float or np.ndarray
            Phase or array of phases of the step normalized to [0, 2π].

        Returns
        -------
        bool or np.ndarray
            Whether adhesion is on for the leg at the given phase(s).
            A boolean array of shape (n,) is returned if ``phase`` is a 1D
            array of n elements; a bool is returned if ``phase`` is a
            scalar.
        """
        swing_start, swing_end = self.swing_period[leg]
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    @property
    def default_pose(self):
        """Default pose of the fly (all legs in neutral position) given as
        a single array. It is ready to be used as the "joints" state in the
        action space of ``NeuroMechFly`` like the following:
        ``NeuroMechFly.step(action={"joints": preprogrammed_steps.default_pose})``.
        """
        return np.concatenate([self.neutral_pos[leg] for leg in self.legs]).ravel()
