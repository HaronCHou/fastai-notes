# fastai-notes
fastai part1 &amp; part2 notes, part1的中文笔记来自其他作者对hiromis笔记的翻译，part2的中文笔记为英文笔记翻译而成

## fastai part1 notes
- https://github.com/hiromis/notes 参考hiromis的笔记，有中文版，质量较高。
  - [chinese](/chinese)文件夹，来源于https://github.com/hiromis/notes 

## fastai part2 notes
- lesson8 markdown 笔记源文件： [lesson8 中文版笔记 md版本](./lesson8/lesson8.md)
- lesson8 csdn 笔记地址：[
fastai 2019 lesson8 notes 笔记_hello world-CSDN博客](https://blog.csdn.net/haronchou/article/details/120541922)
- lesson9 markdown ：[lesson9 中文版笔记 md版本](./lesson9/lesson9.md)
- lessson10 markdown: [lesson10 中文版笔记 md版本](./lesson10/lesson10.md)
- 

## windows vscode ssh

- https://zhuanlan.zhihu.com/p/86637316

- https://docs.microsoft.com/zh-cn/windows-server/administration/openssh/openssh_install_firstuse

  - **步骤1.检查windows本地是否安装有ssh。** 

    windows powershell 管理员模式打开：输入

    `Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'`

- 如果两者均尚未安装，则此操作应返回以下输出：

    ```shell
    Name  : OpenSSH.Client~~~~0.0.1.0
    State : NotPresent
    
    Name  : OpenSSH.Server~~~~0.0.1.0
    State : NotPresent
    ```
    
- 然后，根据需要安装服务器或客户端组件：

    ```shell
    # Install the OpenSSH Client
    Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
    
    # Install the OpenSSH Server
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    ```
    
- 这两者应该都会返回以下输出：

    ```shell
    Path          :
    Online        : True
    RestartNeeded : False
    ```
    
    
    
    ![](Snipaste_2021-10-08_10-42-16.png)

- win+R gpedit.msc 打开本地组策略编辑器

![](Snipaste_2021-10-08_11-01-59.png)

- https://www.cnblogs.com/wolbo/p/11881641.html

![](Snipaste_2021-10-08_11-02-48.png)