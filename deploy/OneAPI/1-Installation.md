# Connect from Windows with OpenSSH

[参考文档](https://devcloud.intel.com/oneapi/documentation/connect-with-ssh-windows-openssh/)

- SSH Connection Configuration

  - 下载 [SSH Key for Linux/MaxOS/Cygwin](https://devcloud.intel.com/oneapi/download_ssh/?asset=ssh_key) ，如 `devcloud-access-key-193322.txt` 

  - 移动到 `C:\Users\ielym\.ssh` 目录下 (windows) 或 `~/.ssh` 目录下 (linux)，并进入该目录

  - 在该目录下创建一个名为 `config` 的文件，没有后缀，并写入下列内容（注意修改 `devcloud-access-key-193322.txt` 的实际路径）：

    ```
    Host devcloud
    User u193322
    IdentityFile ~/.ssh/devcloud-access-key-193322.txt
    ProxyCommand ssh -T -i ~/.ssh/devcloud-access-key-193322.txt guest@ssh.devcloud.intel.com
    ```

  - 

