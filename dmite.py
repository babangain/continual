from dolomite_engine.hf_models import import_from_huggingface

import_from_huggingface(
    pretrained_model_name_or_path="ibm-granite/granite-3b-code-base",
    save_path="dolomite_compatible_model"
)