
"""
Complete staged hybrid model for P642 VTS.

Six ML stages + deterministic timers + four SR flip-flops.
Call step() once per simulation timestep.
"""

import torch

from src.models.conditional_mlp_film import FILM_SETTINGS
from src.models.stage1_model import Stage1MLP, STAGE1_INPUTS, STAGE1_OUTPUTS
from src.models.stage2_model import Stage2MLP, STAGE2_INPUTS, STAGE2_OUTPUTS
from src.models.stage3_model import Stage3MLP, STAGE3_INPUTS, STAGE3_OUTPUTS
from src.models.stage4_model import Stage4MLP, STAGE4_INPUTS, STAGE4_OUTPUTS
from src.models.stage5_model import Stage5MLP, STAGE5_INPUTS, STAGE5_OUTPUTS
from src.models.stage6_model import Stage6MLP, STAGE6_INPUTS, STAGE6_OUTPUTS

from src.timing.timers import PickupDelay, DropoffDelay
from src.timing.flip_flop import SRFlipFlop


class VTSStagedModel:

    def __init__(
        self,
        sample_time_s=0.001,
        timer20_mode="pickup",
        tvts_mode="pickup",
        film_conditioned=False,
    ):
        self.sample_time_s = sample_time_s

        # Six ML layers
        self.film_conditioned = film_conditioned
        stage_kwargs = {"settings_dim": len(FILM_SETTINGS)} if film_conditioned else {}
        self.stage1 = Stage1MLP(**stage_kwargs)
        self.stage2 = Stage2MLP(**stage_kwargs)
        self.stage3 = Stage3MLP(**stage_kwargs)
        self.stage4 = Stage4MLP(**stage_kwargs)
        self.stage5 = Stage5MLP(**stage_kwargs)
        self.stage6 = Stage6MLP(**stage_kwargs)

        for model in [
            self.stage1, self.stage2, self.stage3,
            self.stage4, self.stage5, self.stage6
        ]:
            model.eval()

        # Deterministic timers
        self.timer240 = DropoffDelay(
            delay_s=0.240,
            sample_time_s=sample_time_s
        )

        self.timer20_mode = timer20_mode
        self.tvts_mode = tvts_mode

        self.timer20 = None
        self.tvts_timer = None

        # Four deterministic SR flip-flops
        self.sr1 = SRFlipFlop()
        self.sr2 = SRFlipFlop()
        self.sr3 = SRFlipFlop()
        self.sr4 = SRFlipFlop()

    def reset(self, tvts_delay_s=None):
        """Reset all stateful components before a new case."""

        self.timer240.reset()

        self.timer20 = None
        self.tvts_timer = None

        for flip_flop in [self.sr1, self.sr2, self.sr3, self.sr4]:
            flip_flop.reset()

        if tvts_delay_s is not None:
            self._create_setting_timer(tvts_delay_s)

    def _create_setting_timer(self, delay_s):
        """Create the setting-dependent tVTS timer."""

        timer_class = {
            "pickup": PickupDelay,
            "dropoff": DropoffDelay,
        }.get(self.tvts_mode)

        if timer_class is None:
            raise ValueError("tvts_mode must be 'pickup' or 'dropoff'.")

        self.tvts_timer = timer_class(
            delay_s=float(delay_s),
            sample_time_s=self.sample_time_s
        )

    def _create_20ms_timer(self):
        timer_class = {
            "pickup": PickupDelay,
            "dropoff": DropoffDelay,
        }.get(self.timer20_mode)

        if timer_class is None:
            raise ValueError("timer20_mode must be 'pickup' or 'dropoff'.")

        self.timer20 = timer_class(
            delay_s=0.020,
            sample_time_s=self.sample_time_s
        )

    @staticmethod
    def _predict(model, input_names, values, film_conditioned=True):
        """Run one ML stage and return its binary outputs."""

        missing = [name for name in input_names if name not in values]

        if missing:
            raise KeyError(
                f"Missing inputs for {model.__class__.__name__}: {missing}"
            )

        tensor = torch.tensor(
            [[float(values[name]) for name in input_names]],
            dtype=torch.float32
        )
        settings = None
        if film_conditioned:
            missing_settings = [
                name for name in FILM_SETTINGS if name not in values
            ]
            if missing_settings:
                raise KeyError(
                    f"Missing FiLM settings for {model.__class__.__name__}: "
                    f"{missing_settings}"
                )
            settings = torch.tensor(
                [[float(values[name]) for name in FILM_SETTINGS]],
                dtype=torch.float32,
            )

        with torch.no_grad():
            output = model.predict_binary(tensor, settings)

        return output[0].tolist()

    def _predict_stage(self, model, input_names, values):
        if self.film_conditioned:
            return self._predict(model, input_names, values, True)
        return self._predict(model, input_names, values)

    def step(self, inputs):
        """
        Process one timestep.

        inputs: dictionary containing the external signals and settings.
        Returns all intermediate outputs and the three final outputs.
        """

        # Support the dataset's existing MCB_VTS column name.
        if "MCB_VTS_OPTO" not in inputs and "MCB_VTS" in inputs:
            inputs = {**inputs, "MCB_VTS_OPTO": inputs["MCB_VTS"]}

        detector_aliases = {
            "IA_GT": "IA_OK",
            "IB_GT": "IB_OK",
            "IC_GT": "IC_OK",
            "VAB_GT": "VAB_OK",
            "VBC_GT": "VBC_OK",
            "DELTA_IA_GT": "DELTA_IA_OK",
            "DELTA_IB_GT": "DELTA_IB_OK",
            "DELTA_IC_GT": "DELTA_IC_OK",
        }
        inputs = {
            **inputs,
            **{
                alias: inputs[source]
                for alias, source in detector_aliases.items()
                if alias not in inputs and source in inputs
            },
        }

        if "tVTS" not in inputs:
            raise KeyError("Missing setting-dependent timer input: tVTS")

        if self.tvts_timer is None:
            self._create_setting_timer(inputs["tVTS"])

        # -------------------------------------------------
        # STAGE 1
        # -------------------------------------------------
        s1 = dict(zip(
            STAGE1_OUTPUTS,
            self._predict_stage(self.stage1, STAGE1_INPUTS, inputs)
        ))

        # -------------------------------------------------
        # 240 ms DROP-OFF TIMER
        # -------------------------------------------------
        timer240_out = self.timer240.update(
            bool(inputs["ANY_POLE_DEAD"])
        )

        # -------------------------------------------------
        # STAGE 2
        # -------------------------------------------------
        stage2_inputs = {
            **inputs,
            **s1,
            "240_TIMER_OUT": float(timer240_out),
        }

        s2 = dict(zip(
            STAGE2_OUTPUTS,
            self._predict_stage(self.stage2, STAGE2_INPUTS, stage2_inputs)
        ))

        flip_flop_inputs = {}

        # -------------------------------------------------
        # SR FLIP-FLOP 1
        # S = 5_AND_OUT, R = NOT 4_OR_OUT
        # -------------------------------------------------
        flip_flop_inputs["sr1"] = {
            "set_signal": bool(s2["5_AND_OUT"]),
            "reset_signal": not bool(s2["4_OR_OUT"]),
        }
        sr1_out = self.sr1.next_state(**flip_flop_inputs["sr1"])

        # -------------------------------------------------
        # STAGE 3
        # -------------------------------------------------
        stage3_inputs = {
            **inputs,
            **s1,
            **s2,
            "SR_1_OUT": float(sr1_out),
        }

        s3 = dict(zip(
            STAGE3_OUTPUTS,
            self._predict_stage(self.stage3, STAGE3_INPUTS, stage3_inputs)
        ))

        # -------------------------------------------------
        # SR FLIP-FLOP 2
        # S = 9_AND_OUT
        # R = NOT ANY_VOLTAGE_DEPENDENT_FUNCTION
        # -------------------------------------------------
        flip_flop_inputs["sr2"] = {
            "set_signal": bool(s3["9_AND_OUT"]),
            "reset_signal": not bool(inputs["ANY_VOLTAGE_DEPENDENT_FUNCTION"]),
        }
        sr2_out = self.sr2.next_state(**flip_flop_inputs["sr2"])

        # -------------------------------------------------
        # STAGE 4
        # -------------------------------------------------
        stage4_inputs = {
            **inputs,
            **s1,
            **s2,
            **s3,
            "SR_2_OUT": float(sr2_out),
            "240_TIMER_OUT": float(timer240_out),
        }

        s4 = dict(zip(
            STAGE4_OUTPUTS,
            self._predict_stage(self.stage4, STAGE4_INPUTS, stage4_inputs)
        ))

        # -------------------------------------------------
        # tVTS TIMER: input = 6_AND_OUT
        # -------------------------------------------------
        self.tvts_timer.delay_s = float(inputs["tVTS"])

        tvts_timer_out = self.tvts_timer.update(
            bool(s3["6_AND_OUT"])
        )

        # -------------------------------------------------
        # 20 ms TIMER: input = 10_AND_OUT
        # -------------------------------------------------
        if self.timer20 is None:
            self._create_20ms_timer()

        timer20_out = self.timer20.update(
            bool(s4["10_AND_OUT"])
        )

        # -------------------------------------------------
        # SR FLIP-FLOP 3
        # S = tVTS_TIMER_OUT, R = 7_AND_OUT
        # -------------------------------------------------
        flip_flop_inputs["sr3"] = {
            "set_signal": bool(tvts_timer_out),
            "reset_signal": bool(s3["7_AND_OUT"]),
        }
        sr3_out = self.sr3.next_state(**flip_flop_inputs["sr3"])

        # -------------------------------------------------
        # STAGE 5
        # -------------------------------------------------
        stage5_inputs = {
            **inputs,
            **s1,
            **s2,
            **s3,
            **s4,
            "SR_3_OUT": float(sr3_out),
            "VTS_TIMER_OUT": float(tvts_timer_out),
            "20_TIMER_OUT": float(timer20_out),
        }

        # Stage 5's fifth input is the 20 ms timer output.
        # Its old label 10_TIMER_OUT is corrected here.
        s5 = dict(zip(
            STAGE5_OUTPUTS,
            self._predict_stage(self.stage5, STAGE5_INPUTS, stage5_inputs)
        ))

        # -------------------------------------------------
        # SR FLIP-FLOP 4
        # S = 7_OR_OUT, R = 7_AND_OUT
        # -------------------------------------------------
        flip_flop_inputs["sr4"] = {
            "set_signal": bool(s5["7_OR_OUT"]),
            "reset_signal": bool(s3["7_AND_OUT"]),
        }
        sr4_out = self.sr4.next_state(**flip_flop_inputs["sr4"])

        # -------------------------------------------------
        # FINAL STAGE 6
        # -------------------------------------------------
        stage6_inputs = {
            **inputs,
            **s5,
            "SR_4_OUT": float(sr4_out),
        }

        for name, flip_flop in (
            ("sr1", self.sr1),
            ("sr2", self.sr2),
            ("sr3", self.sr3),
            ("sr4", self.sr4),
        ):
            flip_flop.update(**flip_flop_inputs[name])

        final_outputs = dict(zip(
            STAGE6_OUTPUTS,
            self._predict_stage(self.stage6, STAGE6_INPUTS, stage6_inputs)
        ))

        # Return intermediate and final signals.
        return {
            **s1,
            **s2,
            **s3,
            **s4,
            **s5,
            "240_TIMER_OUT": float(timer240_out),
            "tVTS_TIMER_OUT": float(tvts_timer_out),
            "20_TIMER_OUT": float(timer20_out),
            "SR_1_OUT": float(sr1_out),
            "SR_2_OUT": float(sr2_out),
            "SR_3_OUT": float(sr3_out),
            "SR_4_OUT": float(sr4_out),
            **final_outputs,
        }


if __name__ == "__main__":
    print("P642 VTS staged integration module loaded.")
    print("Six ML stages, three timers and four SR flip-flops.")
