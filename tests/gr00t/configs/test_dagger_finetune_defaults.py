from gr00t.configs.finetune_config import FinetuneConfig


def make_config(**kwargs) -> FinetuneConfig:
    return FinetuneConfig(
        base_model_path="checkpoint",
        dataset_path="dataset",
        embodiment_tag="new_embodiment",
        **kwargs,
    )


def test_dagger_keeps_standard_training_defaults():
    ordinary = make_config()
    dagger = make_config(dagger_expert_only=True)

    assert (ordinary.learning_rate, ordinary.max_steps, ordinary.save_steps) == (
        1e-4,
        10000,
        1000,
    )
    assert (dagger.learning_rate, dagger.max_steps, dagger.save_steps) == (
        1e-4,
        10000,
        1000,
    )
    assert dagger.warmup_ratio == 0.05


def test_dagger_explicit_training_values_take_precedence():
    dagger = make_config(
        dagger_expert_only=True,
        learning_rate=2e-5,
        max_steps=300,
        save_steps=25,
    )

    assert (dagger.learning_rate, dagger.max_steps, dagger.save_steps) == (2e-5, 300, 25)
