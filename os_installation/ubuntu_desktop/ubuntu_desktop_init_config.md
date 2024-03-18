# 安装Chrome

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```



# 安装 Typora

+ 下载 `deb` 包 ：[Typora官网-linux](https://typoraio.cn/#linux)

```bash
sudo apt install ./typora_1.2.3_amd64.deb
```



# Git

```bash
sudo apt install git
git config --global user.name "ielym"
git config --global user.email "ieluoyiming@163.com"
```

+ 添加 ssh key [添加ssh key](https://github.com/ielym/Notes/blob/main/git/10-%E7%94%9F%E6%88%90%E5%AF%86%E9%92%A5.md)



# VS Code

+ 下载 `deb` 包：[VS Code 官网](https://code.visualstudio.com/)

```bash
sudo dpkg -i code_1.70.0-1659589288_amd64.deb
```

+ Python 配置
  + 搜索并安装 Python 插件
  + F1 -> Python: Select Interpreter -> 选择python 解释器
  + File -> Preference -> Settings -> Text Editor -> Font -> 修改 Font Family 为 monospace
  + 字体 14 即可
  + 完成



# Notepad++

```bash
sudo snap install notepad-plus-plus # 等待一会
```



# 有道翻译

+ 下载 deb 包 : [有道翻译官网](http://cidian.youdao.com/multi.html)

  ```python
  sudo apt -f install ./youdao-dict_6.0.0-ubuntu-amd64.deb
  ```

  
