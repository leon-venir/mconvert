# 角色与目标

你是一名精通计算材料科学与 Python 桌面应用开发的软件工程师。我们要新建一个项目 `mconvert`，
它是 `mkits` 的配套 GUI 工具：一个基于 PySide6 的可视化晶体结构文件格式转换器，
最终以**单文件 .exe** 的形式分发给不熟悉 Python 的学生用户，做到双击即用、无需安装
Python/pip/任何依赖。

`mconvert` 不 `pip install mkits` 作为运行时依赖，而是把 `mkits.structure` 这套解析/写出
逻辑**代码级 vendor（拷贝同步）**进 `mconvert` 自己的代码树，成为自包含项目，原因：

1. 单文件 exe 面向零 Python 基础的学生，运行环境里没有 pip、没有网络，打包时越自包含、
   依赖链越短，PyInstaller 出问题的概率越低（尤其 editable install 的包在冻结时容易漏文件）。
2. `mkits` 主仓库仍在快速重构（结构解析之外还有 vasp/qe/workflow 等一堆和格式转换无关的
   模块），`mconvert` 只需要其中一小部分能力，vendor 一份**专属、精简、稳定**的副本，
   不会因为 `mkits` 主线开发而意外破坏面向学生发布的转换工具。
3. 但又不想变成"复制一份从此各改各的"死代码：用同步脚本保留"从 mkits 更新引擎代码"的
   通道，见下方「引擎 vendor 策略」。

## mkits 现有能力（转换引擎来源，已确认）

`mkits/structure.py` 中的 `struct` 类（连同其依赖的 `functions.py`、`database.py`）：

- 输入自动识别格式：`poscar, cif, xsd, xyz, qein, qeout, lammps, wien2k`
- 输出支持格式（`write_struct(calculator=...)`）：`poscar, qein, cif, xyz, lammps, wien2k`
  （`xsd`、`qeout` 只能读、不能写）
- 用法（vendor 后 import 路径变为 `mconvert.engine.structure`，`converter.py` 门面内部这样调用）：
  ```python
  from mconvert.engine.structure import struct
  s = struct("POSCAR")                 # 自动识别输入格式，解析失败抛 MkitsError
  s.write_struct(
      fpath="./", fname="out.cif",
      calculator="cif",                # 目标格式
      dyn=False,                       # 是否写 selective dynamics（仅 poscar/qein 生效）
      frac=True,                       # 分数坐标 or 笛卡尔坐标（仅 poscar/qein 生效）
      write2file=True,                 # False 时返回 (dict, lines) 而不落盘，可用于预览
  )
  ```
- 附加能力（v2 可选）：`get_spacegroup / get_symmetry / get_primitive_cell`（spglib）、
  `supercell`、`apply_strain`、`kmesh`、`seekpath.write_kpath`（能带路径生成）。
- 错误处理：所有已知错误都是 `mkits.functions.MkitsError`，带可读信息，GUI 直接展示即可。

# 项目结构

```
mconvert/
├── design.md                 # 本文档
├── setup.py                  # pip 打包配置（开发/CI 用，不是给学生用的分发形式）
├── requirements.txt
├── README.md
├── build.spec                 # PyInstaller 打包配置（--onefile --windowed + spglib 数据文件）
├── tools/
│   └── sync_engine.py          # 从 ../mkits/mkits/ 同步 structure/functions/database 到
│                                # mconvert/engine/，改写 import 路径，写入来源版本信息
├── mconvert/
│   ├── __init__.py
│   ├── __main__.py            # `python -m mconvert` 入口，创建 QApplication
│   ├── app.py                  # main() 函数：组装 QApplication + MainWindow，全局异常钩子
│   ├── main_window.py           # MainWindow(QMainWindow)：单文件 tab + 批量 tab
│   ├── convert_worker.py        # BatchConvertWorker(QThread)，批量转换时在后台跑
│   ├── batch.py                  # discover_input_files/resolve_output_paths：纯 Python，
│   │                              # 不依赖 PySide6，从 convert_worker.py 拆出来保证在没装
│   │                              # GUI 框架的环境里也能 import/测试
│   ├── brillouin.py             # 第一布里渊区几何（纯 numpy，Wigner-Seitz 构造，3D 多面体 / 2D 多边形）
│   ├── converter.py             # 引擎门面(facade)：GUI 只调用这一层，不直接碰 engine.*
│   ├── formats.py               # 格式清单、默认文件名、哪些格式支持 frac/dyn 选项
│   ├── engine/                  # vendored from mkits，禁止手改；要改先去 mkits 改再同步
│   │   ├── __init__.py
│   │   ├── VENDORED_FROM.txt    # 记录同步来源（mkits git commit hash + 日期）
│   │   ├── structure.py         # 同步自 mkits/mkits/structure.py，import 改为包内相对导入
│   │   ├── functions.py
│   │   └── database.py
│   └── resources/               # 图标等静态资源
└── tests/                          # 一个模块一个测试文件；不需要 PySide6 的（formats/
    ├── test_formats.py              # converter/batch）在系统 Python 下就能跑，需要 PySide6
    ├── test_converter.py            # 的（main_window/convert_worker）在没装 PySide6 的环境
    ├── test_batch.py                # 里通过 unittest.skipUnless 自动跳过而不是报错
    ├── test_main_window.py
    └── test_convert_worker.py
```

## 引擎 vendor 策略（`tools/sync_engine.py`）

- `mkits/structure.py` 里 `from mkits import database` / `from mkits import functions`
  会被同步脚本改写为 `from . import database` / `from . import functions`（因为在
  `mconvert.engine` 包内是相对导入），其余代码逐字节拷贝，不手工修改。
- 每次同步后更新 `engine/VENDORED_FROM.txt`（记录来源 commit + 日期），方便日后追溯
  "这份 structure.py 对应 mkits 主仓库的哪个版本"、判断是否需要重新同步。
- 规则：`mconvert/engine/` 下的文件视为生成产物，不在其中手写业务逻辑或修 bug；
  如果转换过程中发现 `mkits.structure` 的 bug 或缺口，先去 `mkits` 主仓库修，再跑一次
  `sync_engine.py` 同步过来 —— 这样两个项目不会各自分叉出两份互不兼容的解析逻辑，
  也是"为后续开发保留开放性"的关键：mconvert 能持续享受 mkits 结构解析能力的改进，
  而不需要每次手工移植。
- `mconvert/converter.py` 作为唯一调用 `engine.structure.struct` 的门面层：`main_window.py`
  / `convert_worker.py` 只调用 `converter.py` 暴露的函数
  （`load_structure(path) -> LoadedStructure`、
  `preview(loaded, target_format, frac, dyn) -> str`、
  `convert_to_file(loaded, target_format, output_path, frac, dyn)`），
  不直接 import `engine`。`LoadedStructure` 是不透明句柄（内部持有 engine 的 `struct` 实例 +
  一份 `StructureSummary`），GUI 只读 `.summary`，不深入其内部。引擎的 `MkitsError` 在门面层
  被统一转换成 `converter.ConversionError`，GUI 只需要认识这一个异常类型。好处：
  - 未来若要整体替换引擎（比如接入 ASE 做兜底解析），只改 `converter.py` 一处；
  - 未来若要把 GUI 换成别的框架（比如从 PySide6 换 Web），转换逻辑不受影响。

# 依赖

- `PySide6`（GUI 框架）
- `numpy`、`spglib`（`engine/structure.py` 的直接依赖，走正常 pip 依赖，非 vendor）
  - `spglib` 在自己的包目录下放了一份编译好的共享库（`spglib/lib64/libsymspg.so`，
    Windows 上是对应的 `.dll`），运行时按包内相对路径动态加载，PyInstaller 的自动依赖
    扫描并不总能跟踪到这种加载方式。`build.spec` 里没有手写 `hiddenimports`，而是用
    `PyInstaller.utils.hooks.collect_all("spglib")` 把整个已安装的包目录（子模块、
    数据文件、二进制文件）原样收进去——已经在 Linux 上实际打包验证过（打包出的可执行
    文件里 `import spglib; spglib.get_symmetry_dataset(...)` 能正常算出空间群），
    Windows 上的等价性由 `.github/workflows/build-windows.yml` 里的 CI 构建负责验证。
- ~~`mkits`~~：不作为运行时依赖，通过 `tools/sync_engine.py` vendor 到 `mconvert/engine/`。

# 打包目标：Windows 单文件 exe（重要约束）

- **必须在 Windows 上跑 PyInstaller 才能产出 .exe**：PyInstaller 不支持交叉编译，
  在 Linux 上打包只会得到 Linux 可执行文件。你当前的开发环境是 Linux，因此走的是：
  用 GitHub Actions 的 `windows-latest` runner 做 CI 自动打包（`.github/workflows/
  build-windows.yml`：装依赖 → 跑测试 → `pyinstaller build.spec` → 把 `dist/mconvert.exe`
  上传为 workflow artifact）。这个工作流要在 mconvert 被推送到 GitHub 仓库后才会真正触发。
  `build.spec` 本身已经在本机用 Linux 版 PyInstaller 实际跑通过一次，验证了
  `collect_all("spglib")` 的收集逻辑没有问题；CI 跑的是同一份 spec 在 Windows 上的等价构建。
- 面向"不熟悉 Python 的学生"的具体要求：
  - `--windowed`（`build.spec` 里 `console=False`）：不弹黑色控制台窗口；
  - `--onefile`：单个 .exe，双击即用，不需要解压/安装；
  - 未捕获异常一律弹出中文错误对话框（`app.py` 的 `install_exception_hook()`，traceback
    放在可展开的"Show Details"里），绝不能让程序静默崩溃或在没有控制台的情况下直接闪退；
  - 提供了默认图标（`resources/icon.ico` + `icon.png`，几何风格的晶格图案），
    `build.spec` 用作 exe 图标，`app.py` 里 `app.setWindowIcon()` 用作运行时窗口图标。

# GUI 设计（v1：仅格式转换）

## 单文件转换 tab

1. 顶部：选择输入文件按钮 + 拖拽区域，选中后立即用 `struct()` 解析：
   - 成功：显示自动识别出的输入格式（`calculator`）、化学式、原子总数、晶格参数 a/b/c/α/β/γ
   - 失败：以状态栏/提示框展示 `MkitsError` 的错误信息，不允许继续转换
2. 中间：输出设置
   - 目标格式下拉框：仅列出 `write_struct` 支持的输出格式（poscar/qein/cif/xyz/lammps/wien2k）
   - 坐标类型单选：分数 / 笛卡尔 —— 仅当目标格式在 `{poscar, qein}` 时可用，否则置灰并提示
     "该格式坐标写法固定"
   - Selective dynamics 复选框 —— 同样仅 `{poscar, qein}` 可用
   - 输出路径 + 文件名（按目标格式给默认文件名，见下方 `formats.py` 设计）
3. "预览"按钮：调用 `write_struct(write2file=False)` 拿到生成的文本内容，只读文本框展示，
   确认无误后再点"另存为"落盘，避免误覆盖文件
4. 底部：转换按钮 + 状态栏反馈成功/失败

## 批量转换 tab

1. 选择输入文件夹（可选是否递归）+ 文件后缀/内容自动识别，列成表格：
   文件名 | 识别出的格式 | 转换状态（待处理/成功/失败原因）
2. 统一的目标格式与选项设置（同单文件 tab）
3. 输出目录（默认在原目录下新建 `converted/`，不覆盖源文件）
4. "开始批量转换"：在 `QThread`（`convert_worker.py`）里跑，逐文件 try/except，
   单个文件失败不能中断整体队列，实时更新表格状态列
   （必须用后台线程：批量文件多时逐个 `struct()` 解析 + IO 会阻塞主线程卡死 UI）

# formats.py 设计（GUI 与引擎之间的能力映射表）

```python
INPUT_FORMATS = ["poscar", "cif", "xsd", "xyz", "qein", "qeout", "lammps", "wien2k"]  # 仅用于展示，不需用户手选，自动识别
OUTPUT_FORMATS = ["poscar", "qein", "cif", "xyz", "lammps", "wien2k"]
FRAC_CART_TOGGLE_FORMATS = {"poscar", "qein"}
DYN_TOGGLE_FORMATS = {"poscar", "qein"}
DEFAULT_FILENAMES = {
    "poscar": "POSCAR",
    "qein": "pwscf.in",
    "cif": "structure.cif",
    "xyz": "structure.xyz",
    "lammps": "structure.lmp",
    "wien2k": "case.struct",
}
```

# 错误处理原则

- `converter.ConversionError`：预期内的业务错误（格式不支持、解析失败、写文件失败等）。
  GUI 只需要认识这一个异常类型（`converter.py` 内部把引擎抛出的 `MkitsError` 统一转换成它），
  直接展示错误信息，不显示 traceback。
- 其他未捕获异常（从 `converter.py` 里漏出来的，说明是没被转换成 `ConversionError` 的意外
  情况）：说明 vendor 进来的引擎代码有未覆盖的边界情况，弹窗展示 traceback 方便定位；同时
  也应该反馈回 `mkits` 主仓库修复（下次 `sync_engine.py` 时一并带过来），不要在 `mconvert`
  里直接改 `engine/` 下的文件绕过。
- 批量转换中每个文件的异常都要单独捕获记录，不能让一个坏文件中断整个批量任务。

# 严格规则与指南（与 mkits/design.md 保持一致）

1. 语言要求：所有生成的代码注释使用英语，UI 界面文案使用中文。
2. 注释风格：标准、简练的专业注释，禁止对话式风格注释，不要把修改过程写进注释。
3. 分步执行机制：逐个模块推进（先 `tools/sync_engine.py` 把引擎 vendor 进来 → 再
   formats.py → 再 converter.py → 再 main_window.py → 再 convert_worker.py → 再
   app.py/**main**.py → 最后打包配置），未明确指示前不要一次性生成整个项目骨架。
4. `mconvert/engine/` 下的文件只能通过 `sync_engine.py` 更新，不手工编辑；所有 GUI 代码
   只通过 `converter.py` 门面访问引擎，不直接 `import mconvert.engine.*`。

# 里程碑

- **M0** 脚手架：`tools/sync_engine.py` 跑通、把 `structure/functions/database` vendor 进
  `mconvert/engine/` 并能正常 import；`setup.py` + `__main__.py` + 空白 `MainWindow` 能跑起来
- **M1** 单文件转换闭环：选文件 → 自动识别 → 展示结构信息 → 选目标格式/选项 → 预览 → 落盘
- **M2** 批量转换：文件夹扫描、后台线程、结果表、失败重试
- **M3** 打磨与打包 —— 已完成，除了"在真 Windows 上出一份 .exe 验证"这一步（本机是
  Linux，需要 mconvert 推送到 GitHub 后由 `build-windows.yml` 跑一次才能拿到）：
  - 拖拽支持：单文件 tab 拖文件、批量 tab 拖文件夹
  - 全局异常对话框（`app.py`，M0 阶段做的）
  - 界面美化：用 Qt 内置 `Fusion` style 而不是引入 `qt-material`/`qdarkstyle`，
    跨平台观感一致、不增加打包体积和额外的 PyInstaller 兼容性风险
  - 图标（`resources/icon.ico`/`icon.png`）
  - `setup.py` + `build.spec`（`PyInstaller.utils.hooks.collect_all("spglib")`
    处理其动态加载共享库的问题，已在本机 Linux 上打包验证通过）
  - `.github/workflows/build-windows.yml`：CI 里跑测试 + 在 `windows-latest` 上
    实际产出 `mconvert.exe`
- **M4（可选，v2）**：接入引擎里的进阶结构操作 —— 空间群识别、超胞生成、原子替换/添加，
  作为转换前的可选预处理步骤（走 `converter.py` 增加新的门面函数，不影响已有转换流程）

# 实/倒空间 tab（M5，视图已完成，待 Windows 上验证打包）

- 渲染用 PyQtGraph 的 `GLViewWidget`（新依赖：`pyqtgraph`、`PyOpenGL`，加入 `requirements.txt`
  后要在 Windows 上重新验证 PyInstaller 打包）。
- 数据门面（已完成）：`converter.get_real_space_scene(loaded)` 返回原子（序号/元素/名称/原子序数/
  质量/分数与笛卡尔坐标）+ 晶格 + 晶胞 12 条边；`converter.get_reciprocal_scene(loaded)` 返回倒格矢
  （含 2π）+ 第一布里渊区边 + 高对称点（分数/笛卡尔坐标，GAMMA 显示为 Γ）+ 路径段。
  全部是 numpy 数组 + frozen dataclass，渲染层不接触引擎对象。
- 3D 高对称点来自引擎 `seekpath` 的 Setyawan–Curtarolo 标准路径（14 种布拉维格子、含参数相关变体，
  已核对：19 个代表性晶胞下变体识别正确且所有点都落在第一布里渊区内，见 `test_converter.py`）。点的分数坐标
  是相对于输入结构自己的倒格基。输入若不是原胞（如 8 原子的 Si 立方胞），这些点按原胞定义，会落在所画
  （较小）布里渊区之外，对应能带折叠——`ReciprocalScene.points_outside_zone` 标出这种情况，视图里给出提示。
  判定留了 2% 的容差，避免把略有畸变的晶胞（弛豫后、POSCAR 取整）误判成非原胞。
- 2D 路径仍是引擎里的近似表（只区分 4 种二维布拉维格子，无带心矩形）。
- 2D 判定沿用引擎的 `is_2d` 启发式（c 轴真空），2D 时布里渊区是 b1、b2 平面内的多边形。
- 视图（已完成）：`space_viewer.py`（`SpaceViewerTab` + `PickableGLView`）。左侧 `GLViewWidget`
  （拖拽旋转、滚轮缩放、右键平移），右侧实/倒空间切换、显示开关、选中信息面板。
  实空间点击原子显示元素/序号/原子序数/质量/分数与笛卡尔坐标（不画化学键）；倒空间显示布里渊区、
  b1/b2/b3 轴、k 路径与高对称点，点击高对称点显示其分数与笛卡尔坐标。
  结构来源：自带"选择结构文件"/拖拽，或在"单文件转换"里加载后自动同步过来。
- 拾取在屏幕空间做（`picking.py`，纯 numpy）：用当前相机的 view/projection 矩阵把候选点投影到
  屏幕，取点击位置落在其投影圆盘内、且离相机最近的点；不依赖 GPU 回读，可以用合成矩阵单测。
- 注意：`GLTextItem` 默认 `glOptions="additive"`，在白底上深色文字会被加成白色而不可见，
  必须用 `"translucent"`。`PYOPENGL_PLATFORM` 在 Wayland 会话里默认选 EGL，而 Qt 走 xcb/GLX 时
  上下文取不到——用 Qt 的 wayland 平台，或在 X11 下显式 `PYOPENGL_PLATFORM=glx`。
- 无头环境（offscreen 平台）没有 GL 上下文，渲染只能在真实显示器上验证；`tests/test_space_viewer.py`
  只覆盖场景构建、模式切换、经相机矩阵的拾取和信息面板，不覆盖像素输出。
