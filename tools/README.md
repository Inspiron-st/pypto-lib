# pypto-lib tools

## export_all_kernel_insight.py

Exports MindStudio Insight / msprof op-simulator traces for every generated
PTOAS InCore kernel in a `build_output/<case>` directory.

Use it directly on an existing build:

```bash
python tools/export_all_kernel_insight.py --build-dir build_output/Qwen3Decode_<timestamp>
```

For Qwen3-14B decode, the same export can be requested as part of the normal
case run:

```bash
python models/qwen3/14b/qwen3_14b_decode.py --max-seq --enable-l2-swimlane --enable-pmu 2 --export-kernel-insight
```

Output is written under the selected build directory as
`kernel_insight_all_funcs_<timestamp>/`. The file
`latest_all_funcs_kernel_insight_export_root.txt` points to the newest export.
