# Linux Fundamentals for Robotics

## Why Linux for Robotics?

**Linux** is the standard operating system for robotics because:
- **Open source**: Free and customizable
- **ROS2 support**: Best compatibility
- **Performance**: Efficient resource usage
- **Community**: Massive support and tools
- **Real-time capable**: With RT patches

**Recommended Distribution:** Ubuntu 22.04 (for ROS2 Humble)

---

## 1. Basic Navigation

### File System Structure
```
/               Root directory
├── home/       User home directories
│   └── user/   Your files
├── opt/        Optional software (ROS2 installs here)
├── usr/        User programs
├── etc/        Configuration files
├── var/        Variable data (logs)
└── tmp/        Temporary files
```

### Navigation Commands
```bash
pwd             # Print working directory (where am I?)
ls              # List files
ls -la          # List with details and hidden files
cd directory    # Change directory
cd ..           # Go up one level
cd ~            # Go to home directory
cd /            # Go to root
```

**Paths:**
- Absolute: `/home/user/ros2_ws` (starts with /)
- Relative: `../src` (relative to current location)
- Home shortcut: `~/Documents` (~ means /home/user)

---

## 2. File Operations

### Creating & Viewing
```bash
mkdir folder_name       # Create directory
mkdir -p path/to/dir    # Create nested directories

touch file.txt          # Create empty file
nano file.txt           # Edit file (simple editor)
vim file.txt            # Edit file (advanced editor)

cat file.txt            # View entire file
less file.txt           # View file page-by-page (q to quit)
head file.txt           # View first 10 lines
tail file.txt           # View last 10 lines
tail -f log.txt         # Follow file updates (for logs)
```

### Copying & Moving
```bash
cp source.txt dest.txt          # Copy file
cp -r folder/ new_folder/       # Copy directory recursively

mv old_name.txt new_name.txt    # Rename file
mv file.txt /path/to/directory/ # Move file

rm file.txt                     # Delete file
rm -r folder/                   # Delete directory recursively
rm -rf folder/                  # Force delete (careful!)
```

**⚠️ Warning:** `rm` is permanent - no recycle bin!

---

## 3. Permissions

### Understanding Permissions
```bash
ls -l file.txt
# Output: -rw-r--r-- 1 user group 1234 Oct 29 10:00 file.txt
#         ^^^^^^^^^
#         permissions
```

**Permission Format:**
```
-  rw-  r--  r--
│   │    │    │
│   │    │    └─ Others (everyone else)
│   │    └────── Group
│   └─────────── User (owner)
└─────────────── Type (- = file, d = directory)
```

**Permission Letters:**
- `r` = read (4)
- `w` = write (2)
- `x` = execute (1)

### Changing Permissions
```bash
chmod +x script.sh      # Make executable
chmod 755 script.sh     # rwxr-xr-x
chmod 644 file.txt      # rw-r--r--

# Common combinations:
# 755 = rwxr-xr-x (executable files)
# 644 = rw-r--r-- (regular files)
# 777 = rwxrwxrwx (everyone can do everything - avoid!)
```

### Changing Ownership
```bash
sudo chown user:group file.txt
sudo chown -R user:group directory/
```

---

## 4. Package Management (APT)

### Installing Software
```bash
sudo apt update                 # Update package list
sudo apt upgrade                # Upgrade installed packages
sudo apt install package_name   # Install package
sudo apt remove package_name    # Remove package
sudo apt search keyword         # Search for packages
apt list --installed            # List installed packages
```

**Common Packages for Robotics:**
```bash
sudo apt install build-essential  # Compilers
sudo apt install git              # Version control
sudo apt install python3-pip      # Python package manager
sudo apt install vim              # Text editor
sudo apt install htop             # System monitor
```

---

## 5. Processes & System Monitoring

### Viewing Processes
```bash
ps                  # Show your processes
ps aux              # Show all processes
top                 # Interactive process viewer
htop                # Better process viewer (install first)
```

### Managing Processes
```bash
command &           # Run in background
Ctrl+C              # Stop current process
Ctrl+Z              # Pause current process
bg                  # Resume in background
fg                  # Resume in foreground

kill PID            # Terminate process
kill -9 PID         # Force kill
killall process_name # Kill all with name
```

### System Information
```bash
uname -a            # System information
df -h               # Disk usage
du -sh directory/   # Directory size
free -h             # Memory usage
lscpu               # CPU information
```

---

## 6. Text Processing & Searching

### Searching in Files
```bash
grep "pattern" file.txt         # Search in file
grep -r "pattern" directory/    # Recursive search
grep -i "pattern" file.txt      # Case-insensitive

find /path -name "*.txt"        # Find files by name
find /path -type f -mtime -7    # Files modified in last 7 days
```

### Text Manipulation
```bash
cat file1.txt file2.txt > combined.txt   # Combine files
echo "text" > file.txt                   # Overwrite file
echo "text" >> file.txt                  # Append to file

wc file.txt                              # Word count
wc -l file.txt                           # Line count

sort file.txt                            # Sort lines
uniq file.txt                            # Remove duplicates
```

---

## 7. Pipes & Redirection

### Redirection
```bash
command > output.txt        # Redirect output to file (overwrite)
command >> output.txt       # Append output to file
command 2> errors.txt       # Redirect errors
command &> all.txt          # Redirect output and errors

command < input.txt         # Use file as input
```

### Pipes
```bash
# Pass output of one command to another
cat file.txt | grep "error"
ps aux | grep "ros"
ls -l | wc -l                          # Count files

# Chain multiple commands
cat log.txt | grep "ERROR" | sort | uniq
```

---

## 8. Environment Variables

### Viewing Variables
```bash
echo $HOME          # Print HOME variable
echo $PATH          # Print PATH
env                 # List all variables
printenv            # Same as env
```

### Setting Variables
```bash
# Temporary (current terminal only)
export MY_VAR="value"
export PATH=$PATH:/new/path

# Permanent (add to ~/.bashrc)
echo 'export MY_VAR="value"' >> ~/.bashrc
source ~/.bashrc    # Reload configuration
```

**Important for ROS2:**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
```

---

## 9. Networking Basics

### Network Information
```bash
ifconfig                # Network interfaces (may need net-tools)
ip addr                 # Modern alternative
ping google.com         # Test connectivity
nslookup domain.com     # DNS lookup
netstat -tuln           # Show open ports
ss -tuln                # Modern alternative
```

### SSH (Remote Access)
```bash
ssh user@hostname       # Connect to remote machine
ssh user@192.168.1.100  # Connect by IP
scp file.txt user@host:/path/  # Copy file to remote
scp user@host:/path/file.txt .  # Copy from remote
```

---

## 10. Shell Scripting Basics

### Creating a Script
```bash
#!/bin/bash
# This is a comment

echo "Starting robot..."

# Variables
ROBOT_NAME="R2D2"
echo "Robot name: $ROBOT_NAME"

# Conditionals
if [ -f "config.txt" ]; then
    echo "Config found"
else
    echo "Config not found"
fi

# Loops
for i in {1..5}; do
    echo "Loop $i"
done

# Functions
start_robot() {
    echo "Robot started"
}

start_robot
```

### Making Script Executable
```bash
chmod +x script.sh
./script.sh
```

---

## 11. Aliases & Functions

### Creating Aliases
```bash
# Temporary
alias ll='ls -la'
alias ros2_build='cd ~/ros2_ws && colcon build --symlink-install'

# Permanent (add to ~/.bashrc)
echo "alias ll='ls -la'" >> ~/.bashrc
```

### Useful Robotics Aliases
```bash
alias ws='cd ~/ros2_ws'
alias src='cd ~/ros2_ws/src'
alias build='colcon build --symlink-install'
alias source_ros='source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash'
```

---

## 12. .bashrc Configuration

**~/.bashrc** runs every time you open a terminal.

### Essential ROS2 Setup
```bash
# Open .bashrc
nano ~/.bashrc

# Add these lines:
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1

# Aliases
alias ws='cd ~/ros2_ws'
alias build='colcon build --symlink-install'

# Save and reload
source ~/.bashrc
```

---

## 13. Useful Keyboard Shortcuts

```bash
Ctrl+C          # Stop current command
Ctrl+Z          # Pause current command
Ctrl+D          # Exit terminal / EOF
Ctrl+L          # Clear screen

Tab             # Auto-complete
Ctrl+R          # Search command history
Ctrl+A          # Move to start of line
Ctrl+E          # Move to end of line
Ctrl+U          # Delete line before cursor
Ctrl+K          # Delete line after cursor

↑/↓ arrows      # Navigate command history
```

---

## 14. System Maintenance

### Disk Cleanup
```bash
sudo apt autoremove         # Remove unused packages
sudo apt autoclean          # Clean package cache
sudo apt clean              # Remove all cached packages

# Find large files
du -ah /home | sort -rh | head -20
```

### Log Management
```bash
journalctl                  # View system logs
journalctl -f               # Follow logs
journalctl -u service_name  # Logs for specific service

# Clear old logs
sudo journalctl --vacuum-time=7d
```

---

## Essential Linux Concepts Summary

✅ **Must Know:**
1. File navigation (cd, ls, pwd)
2. File operations (cp, mv, rm, mkdir)
3. Viewing files (cat, less, nano)
4. Permissions (chmod, chown)
5. Package management (apt)
6. Environment variables
7. Pipes and redirection
8. Basic text searching (grep, find)

✅ **Important for ROS2:**
- Sourcing setup files
- Setting environment variables
- Using .bashrc for configuration
- Understanding file permissions
- Working with log files

✅ **Nice to Have:**
- Shell scripting basics
- SSH for remote access
- System monitoring (htop, top)
- Process management

---

## Learning Path

1. Practice basic navigation daily
2. Get comfortable with file operations
3. Learn to use nano/vim
4. Master permissions and ownership
5. Understand environment variables
6. Create useful aliases in .bashrc
7. Learn basic shell scripting

**Pro Tip:** Use `man command` to read manual pages (e.g., `man grep`)!
**Use `command --help` for quick help!**

---

## Common Mistakes to Avoid

❌ Using `rm -rf` without double-checking
❌ Running commands as root unnecessarily
❌ Forgetting to source ROS2 setup files
❌ Not backing up important files
❌ Ignoring file permissions errors

✅ Use tab completion to avoid typos
✅ Test dangerous commands in a safe directory first
✅ Keep backups of configuration files
✅ Read error messages carefully
