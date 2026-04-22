# tests/conftest.py
import sys
import warnings
from pathlib import Path

# 1. 获取项目根目录（E:\source\source_code_agent）
# __file__ 是当前文件路径：E:\source\source_code_agent\tests\conftest.py
# parent.parent 就是项目根目录
project_root = Path(__file__).parent.parent
# 2. 将根目录加入 Python 搜索路径
sys.path.insert(0, str(project_root))

# 忽略 google_provider 触发的第三方弃用提示，避免污染测试输出
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"app\.providers\.google_provider",
)