# Contributing Guidelines

感谢您对本项目的兴趣! 我们欢迎任何形式的贡献,包括但不限于提交Bug、Pull Request或改进建议。以下是一些指导原则:

## 代码贡献流程

1. fork 本仓库
2. 新建一个分支: `git checkout -b my-new-feature`
3. 修改代码并通过单元测试
4. 提交变更: `git commit -am '添加一些特性'`
5. 推送到分支: `git push origin my-new-feature`
6. 创建一个新的 Pull Request

## 代码风格

- 遵循 [PEP 8 Python 编码风格指南](https://www.python.org/dev/peps/pep-0008/)
- 使用 [Black](https://github.com/psf/black) 工具自动格式化代码
- 在 `pylintrc` 文件中定义了 Pylint 的配置

## 单元测试

- 新增功能需要编写对应的单元测试
- 测试覆盖率至少达到80%
- 运行全部测试: `tox`

## 提交Pull Request

1. 确保您的代码符合代码风格和单元测试要求
2. 更新 `README.md` 文件以反映新增功能
3. 在Pull Request描述中清晰说明您做出的改变和原因

## 发布版本

项目的发布版本由核心维护者发布,版本号使用 [Semantic Versioning](https://semver.org/)。

## 行为准则

我们致力于提供一个友好、安全和积极的环境,请参阅 [Code of Conduct](CODE_OF_CONDUCT.md) 了解详情。



再次感谢您对本项目的贡献!
 

## 附录：命名约定

为了保障代码的可读性, 并且与 Python 社区保持一致。请遵循 Python 官方推荐的风格指南 [PEP 8 Python 编码风格指南](https://www.python.org/dev/peps/pep-0008/)。

### Python 命名约定 （摘录）:
摘录部分命名规则如下

1. **变量名**
    - 使用小写字母和下划线
    - 例如: `my_variable`, `num_items`

2. **函数名**
    - 使用小写字母和下划线
    - 例如: `my_function`, `calculate_total`

3. **类名**
    - 使用大驼峰命名法(每个单词的首字母大写)
    - 例如: `MyClass`, `StudentManager`

4. **模块名**
    - 使用小写字母和下划线
    - 例如: `my_module.py`, `utils.py`

5. **常量名**
    - 使用大写字母和下划线
    - 例如: `MAX_VALUE`, `PI`

6. **私有属性/方法名**
    - 在前面加上一个下划线
    - 例如: `_internal_value`, `_get_data()`
    - 双下划线前缀(`__`)用于避免与子类中定义的名称冲突

7. **受保护的属性/方法名**
    - 在前面加上一个下划线
    - 例如: `_protected_value`, `_protected_method()`

8. **特殊方法名**
    - 使用双下划线前缀和后缀
    - 例如: `__init__()`, `__str__()`

