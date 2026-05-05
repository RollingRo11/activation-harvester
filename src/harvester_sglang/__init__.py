"""External model package for SGLang that adds activation harvesting.

Loaded by SGLang's ModelRegistry when SGLANG_EXTERNAL_MODEL_PACKAGE points to
this module. Each submodule must export EntryClass = <ForCausalLM-class> and
import-time apply the harvest patches.
"""
