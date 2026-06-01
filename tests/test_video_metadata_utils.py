from video_metadata_utils import build_video_metahub_metadata


def test_video_metadata_omits_full_workflow_payloads():
    payload = build_video_metahub_metadata(
        {
            "positive": "prompt",
            "negative": "negative",
            "width": 640,
            "height": 480,
            "frame_rate": 24,
            "frame_count": 48,
            "metadata_status": "complete",
            "metadata_sources": {"positive": "detected"},
        },
        {"workflow": {"nodes": [{"id": 1}]}, "prompt": {"1": {"class_type": "KSampler"}}},
    )

    assert payload["generator"] == "ComfyUI"
    assert payload["media_type"] == "video"
    assert payload["video"]["duration_seconds"] == 2
    assert payload["metadata_status"] == "complete"
    assert "workflow" not in payload
    assert "prompt_api" not in payload


def test_video_metadata_nulls_defaulted_canonical_fields():
    payload = build_video_metahub_metadata(
        {
            "positive": "",
            "model_name": "",
            "seed": 0,
            "width": 640,
            "height": 480,
            "metadata_sources": {
                "positive": "default",
                "model_name": "default",
                "seed": "default",
            },
        },
        {},
    )

    assert payload["prompt"] is None
    assert payload["model"] is None
    assert payload["seed"] is None
