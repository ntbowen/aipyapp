---
name: usercmd
description: 总结当前任务执行经验并生成可复用的用户命令
modes: [task]
arguments:
  - name: --task-name
    type: str
    required: false
    default: "当前任务"
    help: 任务名称，用于命令描述
  - name: --cmd-name
    type: str
    required: true
    help: 生成的命令名称
---

# 生成任务命令

基于当前对话中完成的任务，请帮我：

1. **总结任务执行经验**：
   - 分析本次对话中完成的主要任务
   - 提取关键步骤和执行逻辑
   - 识别任务中使用的工具和方法
   - 识别任务中的变量，例如可变名称和数量，使生成的命令更通用。

2. **生成可复用的用户命令**：
   - 文件名称：`{{ cmd_name }}.md`
   - 文件目录： `{{ ctx.settings.config_dir }}/commands`
   - 任务描述：{{ task_name }}
   - 命令应该能够完成类似的任务

## 生成要求

请生成一个完整的 Markdown 格式的用户自定义命令文件，包含：

1. **YAML frontmatter 配置**：
   - name: 命令名称
   - description: 命令描述
   - modes: 支持的模式（task 或 main），生成命令时固定使用 `main`
   - arguments: 必要的参数配置，会变成用户输入命令参数，经过argparse解析后成为args全局变量(argparse.Namespace类型)

2. **命令内容主体**：
   - 清晰的任务描述
   - 步骤说明或指令
   - 包含可执行的 Python/Shell 代码块。注意：**必需**使用 ````lang 而不是 ```lang 格式。
   - 期望的输出格式说明
   - 可以使用 Jinja2 格式引用 arguments 定义的参数，例如：{% raw %}`{{ url }}`{% endraw %}

3. **使用示例**：
   - 展示如何调用命令
   - 说明参数的使用方法
   - 预期结果描述

4. **注意事项**:
   - 用户命令里不能使用 `utils` 模块，你需要重构代码去掉对`utils`的使用

## 输出格式

请以 Markdown 代码块的形式输出完整的命令文件内容，并使用Block-Start和Block-End标记包裹代码块，通过path属性指定输出文件。

示例格式：
<!-- Block-Start: {"name": "get_website_title", "path": "{{ ctx.settings.config_dir }}/commands/get_website_title.md"} -->
```markdown
---
name: get_website_title
description: 获取指定网站的标题
modes: [main]
arguments:
  - name: --url
    type: str
    default: "https://www.google.com"
    help: 要获取标题的网站URL
---

# 获取网站标题

请帮我获取以下网站的标题信息：

## 目标网站
{% raw %}{{ url }}{% endraw %}

## 执行任务

使用````lang嵌入执行代码块，此处省略。

## 任务说明

基于上述获取的信息，请：
1. 确认网站标题是否成功获取
2. 分析标题的含义和用途
4. 提供关于网站内容的简要总结

## 使用示例
获取Google网站标题:
`/get_website_title --url="https://www.google.com"`
```
<!-- Block-End: {"name": "get_website_title"} -->
