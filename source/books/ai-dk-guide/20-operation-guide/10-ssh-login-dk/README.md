# SSH登录开发板

有多种方式可以访问开发板。本文以网线方式为例。

## 上电
1. 断电状态下（即开发板没有通电源），插入已经烧录好的 SD 卡。

如果SD卡已经在开发板上，则跳过此步骤。
```alert type=caution
一定要在断电状态下插入或拔出SD卡，否则有大概率导致开发板和SD卡损坏。
```

2. 给开发板插上电源

插上电源后，开发板上的三个绿灯，会逐个点亮。稍等一会后，三个绿灯都亮了，则表示启动成功。


## 连网线

- 网线一端，插入电脑的空闲网口。电脑可以是实验室的台式机，或自带的笔记本。
- 网线另一端，插入开发板的网口。开发板有 2 个网口，上下排列，要插入上面那个网口。
```alert type=tip
电脑如果没有空闲网口，可在电脑 USB 上插 USB转网口 的转接器，以获得一个空闲网口。电脑如有空闲网口，则直接插入网口，不必再用 USB转网口 的转接器。
```
## 设置电脑IP地址

把电脑的 IP 地址，设置为和开发板同一个网段的地址，以便通过网线访问开发板。Windows 电脑因操作系统版本不同，设置步骤和内容略有细微差别。以下为参考步骤：

- 打开：设置 | 网络。找到 USB 网口对应的网络适配器（或者电脑已有网口的网络适配器），修改 IP 地址的相关设置。
- 设置 [DHCP]：手动
- 设置 [IPV4]：ON
- 设置 [IP]：192.168.137.xxx。 xxx 可以是 1 ~ 254（并剔除开发板的 IP 地址），建议是 2 ~ 254。
  - 192.168.137.0。不可用，因为 0 是网络号。
  - 192.168.137.1。有的系统要求设置网关地址（Gateway），否则无法保存。可以填 192.168.137.1，虽然不会真正用到网关。
  - 192.168.137.255。不可用，因为 255 是广播地址。
  - 昇腾/鲲鹏开发板占用的地址：
    - 192.168.137.100。不可用，因为被昇腾开发板占用了。开发板上下排列的2个网口，上面那个网口的 IP 地址固定为 192.168.137.100。
    - 192.168.137.200。不可用，因为被鲲鹏昇腾开发板占用了。
- 设置 [子网掩码]。设置长度或掩码，视操作系统版本不同而不同。
  - 设置 [子网掩码长度]： 24
  - 或者 [子网掩码]： 255.255.255.0
- 设置 [网关]。
  - 视操作系统版本不同而不同。有的要求设置，否则无法保存。
  - 如要求设置，可填写 192.168.137.xxx。xxx在上述可选的取值范围内，并且和本机设置的 IP 地址不同即可，因为网关实际上不用到。比如 192.168.137.1。
- 点击 [保存]

```alert type=important
电脑通常有多个网络（WiFi，以太网1，以太网2，……），要确保修改的网络是连接开发板的那个，不要改错了。
```

## 是否连通

在电脑上启动命令行终端程序（比如 Windows 操作系统的 `cmd`，或者 `powershell`），并在命令行终端上执行 
`ping` 命令：

- `ping 192.168.137.100` （用于昇腾开发板）
- `ping 192.168.137.200` （用于鲲鹏开发板）


如能看到如下信息，则表明电脑和开发板之间的网络是连通的。

``` cmd
~ % ping 192.168.137.100
PING 192.168.137.100 (192.168.137.100): 56 data bytes
64 bytes from 192.168.137.100: icmp_seq=0 ttl=64 time=0.450 ms
64 bytes from 192.168.137.100: icmp_seq=1 ttl=64 time=0.701 ms
64 bytes from 192.168.137.100: icmp_seq=2 ttl=64 time=0.775 ms
64 bytes from 192.168.137.100: icmp_seq=3 ttl=64 time=0.611 ms
^C
--- 192.168.137.100 ping statistics ---
4 packets transmitted, 4 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 0.450/0.634/0.775/0.121 ms                
```


## 通过 ssh 登录开发板

在电脑的命令行终端中执行如下命令登录开发板：
- `ssh root@192.168.137.100`（昇腾开发板，用 root 用户登录）
- `ssh root@192.168.137.200`（鲲鹏开发板，用 root 用户登录）

```alert type=note title="命令说明"
ssh：通过 ssh 登录<br>
root：以开发板上的 root 账号登录<br>
192.168.137.100 或192.168.137.200：开发板 IP 地址。100 是昇腾开发板，200 是鲲鹏开发板
```
```alert type=warning
root 用户权限最大，不当使用时可毁灭整个系统。在今后的工作或科研中，要尽量避免用 root 用户登录。<br>
开发板仅用作开发，即使出现问题也可重新烧录系统恢复。
```

屏幕提示 `root@192.168.137.100's password:` 时，输入密码 `Mind@123`，输入密码完成后按回车键。
```alert type=note
密码输入过程中，屏幕不会有回显，这是正常的，不必担心。
```

当输入正确密码后，就可以登录开发板，并看到如下信息。
```
~ % ssh root@192.168.137.100
root@192.168.137.100's password: 
    _                                _             _               _     _  _
   / \    ___   ___  ___  _ __    __| |         __| |  ___ __   __| | __(_)| |_
  / _ \  / __| / __|/ _ \| '_ \  / _` | _____  / _` | / _ \\ \ / /| |/ /| || __|
 / ___ \ \__ \| (__|  __/| | | || (_| ||_____|| (_| ||  __/ \ V / |   < | || |_
/_/   \_\|___/ \___|\___||_| |_| \__,_|        \__,_| \___|  \_/  |_|\_\|_| \__|
                                                                                            
Welcome to Atlas 200I DK A2
This system is based on Ubuntu 22.04 LTS (GNU/Linux 5.10.0+ aarch64)

This system is only applicable to individual developers and cannot be used for commercial purposes.
            
By using this system, you have agreed to the Huawei Software License Agreement.
Please refer to the agreement for details on https://www.hiascend.com/software/protocol
            
Reference resources
* Home page: https://www.hiascend.com/hardware/developer-kit-a2
* Documentation: https://www.hiascend.com/hardware/developer-kit-a2/resource
* Online courses: https://www.hiascend.com/edu/courses
* Online experiments: https://www.hiascend.com/zh/edu/experiment
* Forum: https://www.hiascend.com/forum/
* Code: https://gitee.com/HUAWEI-ASCEND/ascend-devkit


The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

root@@davinci-mini:~$                 
```        
