"""调研 Pipeline 的共享状态。"""

from harness import DiscussionOutput, State


class ResearchState(State):
    """Pipeline 各步骤通过 output_key 写入对应字段。"""

    # 输入
    raw_input: str = ""                    # 用户原始输入
    input_type: str = ""                   # "project" | "comparison" | "topic"

    # 调研目标
    target_project: str = ""               # 主调研项目名
    target_url: str = ""                   # 主项目 GitHub URL
    competitors: list[dict] = []           # [{name, url, description}]

    # 采集数据
    github_metrics: dict = {}              # {project_name: {stars, forks, ...}}
    qualitative_info: str = ""             # 定性分析文本

    # Discussion 结果
    discussion: DiscussionOutput | None = None

    # 最终报告
    report: str = ""                       # 最终 Markdown 报告
    output_path: str = ""                  # 写入 SecondBrain 的文件路径
