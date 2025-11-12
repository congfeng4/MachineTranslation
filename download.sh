pip install huggingface-hub
huggingface-cli download wmt17 --repo-type dataset --local-dir ./wmt17_zh-en

huggingface-cli download wmt17 --repo-type dataset \
  --local-dir ./wmt17_zh-en \
  --include "*zh-en*" \
  --exclude "*de-en*" --exclude "*fr-en*" --exclude "*cs-en*" --exclude "*ru-en*" \
  --exclude "*fi-en*" --exclude "*tr-en*" --exclude "*lv-en*"