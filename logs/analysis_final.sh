bash logs/queue_A.sh llava-onevision-7b llavaov7b compact  # 2026-11-26 수동, CPU
python -m src.attn_theory analyze --tags qwen3b qwen7b qwen3vl8b internvl8b llavaov7b
python -m src.novelty_eval B C D E F
CUDA_VISIBLE_DEVICES=0 bash logs/queue_novelty.sh qwen2.5-vl-7b qwen7b F10   # 2026-11-27 재실행: λ=0.3 이 base 와 같아 강한 마스킹 필요
CUDA_VISIBLE_DEVICES=1 bash logs/queue_novelty.sh qwen2.5-vl-3b qwen3b F10   # 2026-11-27 재실행: λ=0.3 이 base 와 같아 강한 마스킹 필요
python -m src.agg_multi --out_dir results/multi > results/multi_summary.txt 2>/dev/null  (경고는 stderr 로 버림)
python -m src.novelty_figs   # F10 끝난 뒤 다시
python -m src.novelty_eval F > results/novelty/F_report.txt   # λ=1.0 추가 후
