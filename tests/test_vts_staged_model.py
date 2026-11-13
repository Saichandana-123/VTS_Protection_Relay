from types import MethodType

from src.models.vts_staged_model import VTSStagedModel


def test_staged_model_captures_all_sr_inputs_before_any_update():
    model = VTSStagedModel()
    events = []

    def fake_predict(stage, input_names, values):
        outputs = {
            "Stage1MLP": [False, False, False, False, False],
            "Stage2MLP": [False, False, True, False],
            "Stage3MLP": [False, False, False, True],
            "Stage4MLP": [False],
            "Stage5MLP": [False, False, True],
            "Stage6MLP": [False, False, False],
        }
        return outputs[stage.__class__.__name__]

    model._predict = fake_predict
    for name in ("sr1", "sr2", "sr3", "sr4"):
        flip_flop = getattr(model, name)
        original_next_state = flip_flop.next_state
        original_update = flip_flop.update

        def capture_next(
            self,
            set_signal,
            reset_signal,
            *,
            name=name,
            original_next_state=original_next_state,
        ):
            events.append(("capture", name, set_signal, reset_signal))
            return original_next_state(set_signal, reset_signal)

        def commit(
            self,
            set_signal,
            reset_signal,
            *,
            name=name,
            original_update=original_update,
        ):
            events.append(("update", name))
            return original_update(set_signal, reset_signal)

        flip_flop.next_state = MethodType(capture_next, flip_flop)
        flip_flop.update = MethodType(commit, flip_flop)

    inputs = {
        "tVTS": 1.0,
        "ANY_POLE_DEAD": False,
        "ANY_VOLTAGE_DEPENDENT_FUNCTION": True,
        "MCB_VTS_OPTO": False,
    }
    inputs.update(
        {
            name: False
            for name in (
                "IA_OK",
                "IB_OK",
                "IC_OK",
                "DELTA_IA_OK",
                "DELTA_IB_OK",
                "DELTA_IC_OK",
                "VTS_MANRESET",
                "VTS_AUTORESET",
                "VAB_OK",
                "VBC_OK",
                "VTS_BLOCKING",
                "ACCELERATE_IND",
            )
        }
    )

    result = model.step(inputs)

    captures = [event for event in events if event[0] == "capture"]
    updates = [event for event in events if event[0] == "update"]
    first_update = next(
        index for index, event in enumerate(events) if event[0] == "update"
    )
    pre_commit_captures = [
        event for event in events[:first_update] if event[0] == "capture"
    ]
    assert [event[1] for event in pre_commit_captures] == [
        "sr1",
        "sr2",
        "sr3",
        "sr4",
    ]
    assert [event[1] for event in updates] == ["sr1", "sr2", "sr3", "sr4"]
    assert first_update > 0
    assert [result[f"SR_{index}_OUT"] for index in range(1, 5)] == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
