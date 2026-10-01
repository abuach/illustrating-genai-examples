"""
genai — companion library for Programming Generative AI
--------------------------------------------------------
Import anything you need:

    from genai import ask, embed, similarity, DocumentStore, run_agent, think

All LLM calls enforce concise output by default (BRIEF system prompt +
max_tokens=200). Override per-call when you need longer responses:

    ask("Write a 500-word essay...", max_tokens=800, system=None)

Full API by module:
  llm      : ask, chat              (brevity-enforced by default)
  embed    : embed, similarity, semantic_search, word_analogy
  rag      : DocumentStore, chunk, rag
  code     : embed_code, code_similarity, code_search, code_analogy
  vision   : ask_image, ask_images
  imagegen : generate_image       (text-to-image via diffusion)
  audio    : transcribe, transcribe_timed, log_mel, read_wav, asr_size (Whisper)
  privacy  : add_noise, private_avg, federated_avg
  perf     : time_call, time_chat, compare_models
  tokens   : tokenize, count_tokens
  agent    : run_agent              (tool-calling loop)
  thinking : think                  (deepseek-r1 / reasoning models)
  viz      : plot_attention, plot_embeddings_2d
"""

from genai.llm      import (ask, chat, next_token_distribution,
                            BRIEF, DEFAULT_MODEL, DEFAULT_MAX_TOKENS)
from genai.embed    import (embed, similarity, semantic_search, word_analogy,
                            CROSSLINGUAL_STUDY)
from genai.rag      import DocumentStore, chunk, rag, show_grounded
from genai.code     import (embed_code, embed_codebert, embed_codesearch,
                            code_similarity, code_search, code_analogy,
                            code_search_scores, CODER_OUTPUTS, CODE_TASKS,
                            CODE_COLUMNS, grade_code, score_scorecard,
                            show_failure, grade_is_prime)
from genai.vision   import ask_image, ask_images, mnist_digits
from genai.imagegen import generate_image, gallery
from genai.audio    import transcribe, transcribe_timed, log_mel, read_wav, asr_size
from genai.privacy  import add_noise, private_avg, federated_avg, hospital_bp_cohorts
from genai.security import (sanitize, show_turn, show_system_prompt,
                            ask_naive, ask_hardened, ask_about_note,
                            SECURITY_POLICY, REPEAT_ATTACK, THIRD_PERSON_ATTACK,
                            DIRECT_ASK, PREFIX_JAILBREAK, CLEAN_NOTE, POISONED_NOTE,
                            memorization_probe, show_recitation, embedding_leak,
                            show_embedding_attack, poisoning_demo, show_poisoning,
                            redact, Role, User,
                            dp_budget_table, k_anonymize,
                            moderate, show_moderation, watermark_generate,
                            watermark_detect, watermark_demo, show_watermark,
                            audit, GuardedAssistant,
                            CAPSTONE_ATTACK, CAPSTONE_DRAFT)
from genai.perf     import time_call, time_chat, compare_models
from genai.tokens   import (tokenize, count_tokens, token_ids,
                            special_tokens, learn_bpe)
from genai.agent    import (run_agent, run_with_trace, tool_spec, safe_eval, sophia_tools,
                            ask_sophia, show_tool_call,
                            try_step, try_failing_fetch, show_recovery, show_write_safety,
                            compress, show_compression, Memory, World,
                            demo_memory, demo_world, show_world, show_memory,
                            self_refine, show_self_refine, confident_answer, show_confidence,
                            show_sandbox,
                            gate, show_gate, inspect, show_inspect,
                            research_pipeline, show_research, parallel_research,
                            Tracer, route, routed_answer, route_tier, resolve,
                            score_routers, show_routing, show_savings,
                            Sophia,
                            SkillLibrary, write_skill, show_skill_write,
                            show_skill_reuse, skill_retrieval_study,
                            MemoryStream, rate_importance, synthesize_insight,
                            build_sophia_stream, show_importance,
                            show_memory_retrieval, show_memory_reflection,
                            SOPHIA_OBSERVATIONS, SOPHIA_QUERY, SOPHIA_INSIGHT,
                            SOPHIA_STUDENT,
                            AGENT_BENCH, SOPHIA_HISTORY_DEMO, PARALLEL_RESEARCH_TOPICS,
                            ROUTING_TASKS, ROUTING_SOLVE, ROUTER_PICKS,
                            ROUTER_STUDY, ROUTING_SAVINGS, SOPHIA_IDENTITY)
from genai.thinking import (think, snap_answer, think_answer,
                            accuracy_trials, time_models, NOVEL_PROBLEMS,
                            cot_sample, self_consistent, show_self_consistency,
                            SELF_CONSISTENCY_STUDY, PHI4_REASONING_STUDY)
from genai.prompting import (direct_answer, step_back, show_step_back,
                            STEP_BACK_PROBLEMS, STEP_BACK_STUDY)
from genai.pal       import (run_program, cot_answer, pal_solve,
                            show_pal_prose, show_pal, PAL_PROBLEMS, PAL_STUDY)
from genai.cove      import (cove, draft_answer, verify_claim, list_precision,
                            list_recall, show_cove, COVE_QUESTIONS, COVE_STUDY)
from genai.crag      import (web_search, grade_retrieval, refine, crag, standard_rag,
                            answer_hit, show_crag, CRAG_QUERIES, CRAG_STUDY)
from genai.alignment import (PREFERENCE_PAIRS, EVAL_TASKS, SYCOPHANCY_PROBES,
                            ANCHOR_PROBES, CONSTITUTION, BRIEF, gold, fit_reward_model,
                            loo_accuracy, best_of_n, overoptimization_sweep, reward_blindspot,
                            constitution_reward, flip_study,
                            show_judge_gap, show_preferences, show_reward_model,
                            show_best_of_n, show_blindspot, show_sycophancy,
                            show_anchoring, show_constitution)
from genai.research import (RESEARCH_QUESTION, decompose, search, supported,
                            naive_research, gather_round, deep_research,
                            grounding_study, round_coverage,
                            show_naive_report, show_search, show_plan,
                            show_gather, show_verification, show_report)
from genai.viz      import (plot_attention, plot_embeddings_2d,
                            plot_agentic_loop, plot_tool_bench,
                            plot_agent_landscape,
                            plot_rag_pipeline, plot_chunk_size,
                            plot_embedding_space, plot_retrieval_comparison,
                            plot_precision_recall, plot_rag_architecture,
                            plot_code_embeddings,
                            plot_search_comparison, plot_model_speed,
                            plot_size_vs_speed, plot_code_scorecard,
                            plot_privacy_budget,
                            plot_watermark_detection,
                            plot_word_vs_embed_similarity, plot_crosslingual,
                            plot_toy_embeddings,
                            plot_tsne_embeddings, plot_next_token,
                            plot_token_pipeline, plot_multilingual_tokens,
                            plot_token_boundaries, plot_special_tokens,
                            plot_digit_chunking,
                            plot_throughput, plot_context_cost,
                            plot_conversation_cache, plot_generation_sag,
                            plot_prompt_compression, plot_task_ladder,
                            plot_quantization, plot_sparsity, plot_effort_cost,
                            plot_embed_vs_gen, plot_three_pass,
                            plot_thinking_latency, plot_thinking_accuracy,
                            plot_self_consistency, plot_step_back, plot_pal,
                            plot_cove, plot_crag, plot_skill_retrieval,
                            plot_overoptimization, plot_reward_blindspot, plot_sycophancy,
                            plot_research_grounding, plot_research_rounds,
                            BLUE, GREEN, ORANGE, RED, LGRAY, MGRAY, DGRAY)

__all__ = [
    # llm
    "ask", "chat", "next_token_distribution",
    "BRIEF", "DEFAULT_MODEL", "DEFAULT_MAX_TOKENS",
    # embed
    "embed", "similarity", "semantic_search", "word_analogy", "CROSSLINGUAL_STUDY",
    # rag
    "DocumentStore", "chunk", "rag", "show_grounded",
    # code
    "embed_code", "embed_codebert", "embed_codesearch", "code_similarity",
    "code_search", "code_analogy", "code_search_scores",
    "CODER_OUTPUTS", "CODE_TASKS", "CODE_COLUMNS", "grade_code",
    "score_scorecard", "show_failure", "grade_is_prime",
    # vision
    "ask_image", "ask_images", "mnist_digits",
    # imagegen
    "generate_image", "gallery",
    # audio
    "transcribe", "transcribe_timed", "log_mel", "read_wav", "asr_size",
    # privacy
    "add_noise", "private_avg", "federated_avg", "hospital_bp_cohorts",
    # security
    "sanitize", "show_turn", "show_system_prompt", "ask_naive",
    "ask_hardened", "ask_about_note", "SECURITY_POLICY",
    "REPEAT_ATTACK", "THIRD_PERSON_ATTACK", "DIRECT_ASK", "PREFIX_JAILBREAK", "CLEAN_NOTE",
    "POISONED_NOTE", "memorization_probe", "show_recitation", "embedding_leak",
    "show_embedding_attack", "poisoning_demo", "show_poisoning",
    "redact", "Role", "User", "dp_budget_table",
    "k_anonymize", "moderate", "show_moderation", "watermark_generate",
    "watermark_detect", "watermark_demo", "show_watermark", "audit",
    "GuardedAssistant", "CAPSTONE_ATTACK", "CAPSTONE_DRAFT",
    # perf
    "time_call", "time_chat", "compare_models",
    # tokens
    "tokenize", "count_tokens", "token_ids", "special_tokens", "learn_bpe",
    # agent
    "run_agent", "run_with_trace", "tool_spec", "safe_eval", "sophia_tools",
    "ask_sophia", "show_tool_call",
    "try_step", "try_failing_fetch", "show_recovery", "show_write_safety",
    "compress", "show_compression", "Memory", "World",
    "demo_memory", "demo_world", "show_world", "show_memory",
    "self_refine", "show_self_refine", "confident_answer", "show_confidence",
    "show_sandbox", "gate", "show_gate", "inspect", "show_inspect",
    "research_pipeline", "show_research", "parallel_research",
    "Tracer", "route", "routed_answer", "route_tier", "resolve",
    "score_routers", "show_routing", "show_savings",
    "Sophia",
    "SkillLibrary", "write_skill", "show_skill_write", "show_skill_reuse",
    "skill_retrieval_study",
    "MemoryStream", "rate_importance", "synthesize_insight", "build_sophia_stream",
    "show_importance", "show_memory_retrieval", "show_memory_reflection",
    "SOPHIA_OBSERVATIONS", "SOPHIA_QUERY", "SOPHIA_INSIGHT", "SOPHIA_STUDENT",
    "AGENT_BENCH", "SOPHIA_HISTORY_DEMO", "PARALLEL_RESEARCH_TOPICS",
    "ROUTING_TASKS", "ROUTING_SOLVE", "ROUTER_PICKS",
    "ROUTER_STUDY", "ROUTING_SAVINGS", "SOPHIA_IDENTITY",
    # thinking
    "think", "snap_answer", "think_answer",
    "accuracy_trials", "time_models", "NOVEL_PROBLEMS",
    "cot_sample", "self_consistent", "show_self_consistency",
    "SELF_CONSISTENCY_STUDY", "PHI4_REASONING_STUDY",
    # prompting
    "direct_answer", "step_back", "show_step_back",
    "STEP_BACK_PROBLEMS", "STEP_BACK_STUDY",
    # pal
    "run_program", "cot_answer", "pal_solve",
    "show_pal_prose", "show_pal", "PAL_PROBLEMS", "PAL_STUDY",
    # cove
    "cove", "draft_answer", "verify_claim", "list_precision", "list_recall",
    "show_cove", "COVE_QUESTIONS", "COVE_STUDY",
    # crag
    "web_search", "grade_retrieval", "refine", "crag", "standard_rag",
    "answer_hit", "show_crag", "CRAG_QUERIES", "CRAG_STUDY",
    # alignment
    "PREFERENCE_PAIRS", "EVAL_TASKS", "SYCOPHANCY_PROBES", "ANCHOR_PROBES",
    "CONSTITUTION", "BRIEF", "gold", "fit_reward_model", "loo_accuracy",
    "best_of_n", "overoptimization_sweep", "reward_blindspot",
    "constitution_reward", "flip_study",
    "show_judge_gap", "show_preferences", "show_reward_model", "show_best_of_n",
    "show_blindspot", "show_sycophancy", "show_anchoring", "show_constitution",
    # research
    "RESEARCH_QUESTION", "decompose", "search", "supported", "naive_research",
    "gather_round", "deep_research", "grounding_study", "round_coverage",
    "show_naive_report", "show_search", "show_plan", "show_gather",
    "show_verification", "show_report",
    # viz
    "plot_attention", "plot_embeddings_2d",
    "plot_agentic_loop", "plot_tool_bench",
    "plot_agent_landscape",
    "plot_rag_pipeline", "plot_chunk_size", "plot_embedding_space",
    "plot_retrieval_comparison", "plot_precision_recall", "plot_rag_architecture",
    "plot_code_embeddings", "plot_search_comparison", "plot_model_speed",
    "plot_size_vs_speed", "plot_code_scorecard",
    "plot_privacy_budget", "plot_watermark_detection",
    "plot_word_vs_embed_similarity", "plot_crosslingual", "plot_toy_embeddings", "plot_tsne_embeddings",
    "plot_next_token", "plot_token_pipeline", "plot_multilingual_tokens",
    "plot_token_boundaries",
    "plot_special_tokens",
    "plot_digit_chunking",
    "plot_throughput", "plot_context_cost", "plot_conversation_cache",
    "plot_generation_sag",
    "plot_prompt_compression",
    "plot_task_ladder", "plot_quantization", "plot_sparsity", "plot_effort_cost",
    "plot_embed_vs_gen", "plot_three_pass",
    "plot_thinking_latency", "plot_thinking_accuracy", "plot_self_consistency",
    "plot_step_back", "plot_pal", "plot_cove", "plot_crag", "plot_skill_retrieval",
    "plot_overoptimization", "plot_reward_blindspot", "plot_sycophancy",
    "plot_research_grounding", "plot_research_rounds",
    "BLUE", "GREEN", "ORANGE", "RED", "LGRAY", "MGRAY", "DGRAY",
]
