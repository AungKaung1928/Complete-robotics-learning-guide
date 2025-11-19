# Git, Version Control & CI/CD for Robotics Engineers

## Table of Contents
- [Why This Matters in Robotics](#why-this-matters-in-robotics)
- [Git Fundamentals](#git-fundamentals)
- [Version Control Best Practices](#version-control-best-practices)
- [Branching Strategies](#branching-strategies)
- [Git in ROS2 Workspaces](#git-in-ros2-workspaces)
- [CI/CD for Robotics](#cicd-for-robotics)
- [Real-World Scenarios](#real-world-scenarios)

---

## Why This Matters in Robotics

**The Reality:**
- You break a working navigation stack at 2 AM before a demo
- Multiple engineers modify the same MoveIt config
- Your robot works on your laptop but crashes on the actual hardware
- A sensor driver update breaks your entire perception pipeline

**Git + CI/CD solves these problems.**

---

## Git Fundamentals

### Core Concept: Snapshots, Not Deltas

Git doesn't store "file changes" — it stores **complete snapshots** of your project at each commit.

```
Commit A: [navigation_params.yaml v1] [map.yaml v1] [launch file v1]
         ↓
Commit B: [navigation_params.yaml v2] [map.yaml v1] [launch file v1]
         ↓
Commit C: [navigation_params.yaml v2] [map.yaml v2] [launch file v2]
```

### Essential Commands (With Robotics Context)

```bash
# Initialize a ROS2 package as a git repo
cd ~/ros2_ws/src/my_robot_navigation
git init

# Check what changed (ALWAYS do this before committing)
git status
git diff config/nav2_params.yaml

# Stage specific changes (surgical commits, not "git add .")
git add config/nav2_params.yaml
git add launch/navigation_launch.py

# Commit with meaningful message
git commit -m "fix: reduce DWB min_vel_x to prevent oscillation near goals

- Changed min_vel_x from 0.0 to -0.1 m/s
- Tested on TurtleBot3 in narrow corridor scenario
- Resolves issue #42"

# View commit history with graph
git log --oneline --graph --all

# Undo uncommitted changes (DANGEROUS - no undo!)
git checkout -- config/nav2_params.yaml

# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit AND discard changes (VERY DANGEROUS)
git reset --hard HEAD~1
```

### The Three States

```
Working Directory  →  Staging Area  →  Git Repository
(your files)         (git add)         (git commit)

Example:
nav2_params.yaml   →  nav2_params.yaml  →  Commit abc123
(modified)            (staged)             (permanent)
```

---

## Version Control Best Practices

### 1. Atomic Commits (One Logical Change Per Commit)

**❌ Bad:**
```bash
git commit -m "Fixed stuff and added new feature"
# Changed: nav2_params.yaml, costmap_params.yaml, new sensor driver, launch files
```

**✅ Good:**
```bash
git commit -m "fix: increase costmap inflation radius for TurtleBot3"
# Changed: costmap_params.yaml (inflation_radius: 0.55 → 0.65)

git commit -m "feat: add RealSense D435 depth camera driver"
# Changed: launch/sensors_launch.py, config/camera_params.yaml
```

### 2. Commit Message Format (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types for robotics:**
- `feat`: New feature (new sensor integration, new behavior)
- `fix`: Bug fix (parameter tuning, logic error)
- `refactor`: Code restructure (no behavior change)
- `perf`: Performance improvement
- `test`: Add/modify tests
- `docs`: Documentation only
- `ci`: CI/CD pipeline changes
- `config`: Configuration file changes

**Example:**
```bash
git commit -m "fix(navigation): prevent goal oscillation in narrow spaces

- Reduced DWB controller min_vel_x from 0.0 to -0.1 m/s
- Increased xy_goal_tolerance from 0.05 to 0.10 meters
- Tested in 1.2m wide corridor with 0.35m robot base

Resolves: #42
Tested-on: TurtleBot3 Burger (ROS2 Humble)"
```

### 3. What NOT to Commit

Create `.gitignore` file:

```gitignore
# ROS2 Build Artifacts
build/
install/
log/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# C++
*.o
*.so
*.a
*.out

# IDE
.vscode/
.idea/
*.swp

# Bags and logs (too large!)
*.bag
*.db3
*.mcap

# Maps (commit small test maps, not full warehouse scans)
maps/*_large.yaml
maps/*_large.pgm

# Model files (use Git LFS or external storage)
*.onnx
*.pt
*.pth
models/

# Temporary files
*~
.DS_Store
```

**Exception:** DO commit:
- Launch files
- Config files (YAML, URDF, SDF)
- Small test maps
- Package.xml, CMakeLists.txt
- README, documentation

---

## Branching Strategies

### Git Flow for Robotics Teams

```
main (production-ready, runs on actual robot)
  ├── develop (integration branch)
      ├── feature/add-lidar-obstacle-detection
      ├── feature/tune-pid-controller
      ├── fix/amcl-particle-depletion
      └── hotfix/emergency-stop-bug
```

### Commands:

```bash
# Create and switch to feature branch
git checkout -b feature/add-lidar-obstacle-detection

# Work on feature...
git add src/obstacle_detector.cpp
git commit -m "feat(perception): add 2D lidar obstacle clustering"

# Switch back to develop
git checkout develop

# Merge feature (after testing!)
git merge feature/add-lidar-obstacle-detection

# Delete merged branch
git branch -d feature/add-lidar-obstacle-detection
```

### Real Scenario: Emergency Hotfix

```bash
# You're on develop, but main branch has critical bug on robot!
git checkout main
git checkout -b hotfix/emergency-stop-not-working

# Fix the bug
git add src/safety_controller.cpp
git commit -m "fix(safety): emergency stop now preempts all goals

- Added action server preemption in e-stop callback
- Tested: stops within 0.2s on hardware"

# Merge to main AND develop
git checkout main
git merge hotfix/emergency-stop-not-working
git checkout develop
git merge hotfix/emergency-stop-not-working

# Delete hotfix branch
git branch -d hotfix/emergency-stop-not-working
```

---

## Git in ROS2 Workspaces

### Scenario: Multi-Package Workspace

```
ros2_ws/
├── src/
│   ├── my_robot_description/    (Git Repo 1)
│   │   ├── .git/
│   │   ├── urdf/
│   │   └── meshes/
│   ├── my_robot_navigation/     (Git Repo 2)
│   │   ├── .git/
│   │   ├── config/
│   │   └── launch/
│   └── my_robot_perception/     (Git Repo 3)
│       ├── .git/
│       └── src/
└── (build/, install/, log/ - NOT in git)
```

**DO NOT** make `ros2_ws/` a git repo. Each **package** is its own repo.

### Using Git Submodules (For Dependencies)

If your navigation package depends on a custom costmap plugin:

```bash
cd ~/ros2_ws/src/my_robot_navigation
git submodule add https://github.com/company/custom_costmap_plugin.git plugins/custom_costmap

# Clone a repo with submodules
git clone --recursive https://github.com/yourname/my_robot_navigation.git

# Update submodules
git submodule update --remote --merge
```

### Handling Large Files (Maps, Models)

**Option 1: Git LFS (Large File Storage)**

```bash
# Install Git LFS
sudo apt install git-lfs
git lfs install

# Track large files
git lfs track "maps/*.pgm"
git lfs track "models/*.onnx"

# Commit .gitattributes
git add .gitattributes
git commit -m "chore: add Git LFS for maps and models"

# Now commit large files normally
git add maps/warehouse.pgm
git commit -m "feat(maps): add warehouse map for testing"
```

**Option 2: External Storage + Download Script**

```bash
# In your repo, create a script
# scripts/download_assets.sh

#!/bin/bash
wget https://company-storage.com/models/yolov5.onnx -P models/
wget https://company-storage.com/maps/warehouse.pgm -P maps/
```

---

## CI/CD for Robotics

### Why CI/CD Matters

**Without CI/CD:**
- "It works on my machine" syndrome
- Breaking changes discovered during robot testing
- Manual testing wastes expensive robot time

**With CI/CD:**
- Every commit is automatically tested
- Catch integration issues before hardware deployment
- Run simulations in parallel for faster validation

### GitHub Actions for ROS2

Create `.github/workflows/ros2_ci.yml`:

```yaml
name: ROS2 CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build-and-test:
    runs-on: ubuntu-22.04
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Setup ROS2 Humble
      uses: ros-tooling/setup-ros@v0.6
      with:
        required-ros-distributions: humble
        
    - name: Install dependencies
      run: |
        sudo apt update
        rosdep update
        rosdep install --from-paths src --ignore-src -r -y
        
    - name: Build workspace
      run: |
        source /opt/ros/humble/setup.bash
        colcon build --symlink-install
        
    - name: Run tests
      run: |
        source /opt/ros/humble/setup.bash
        source install/setup.bash
        colcon test
        colcon test-result --verbose
```

### Linting and Code Quality

Add to `.github/workflows/lint.yml`:

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-22.04
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run ament_lint
      run: |
        sudo apt update
        sudo apt install -y python3-pip
        pip3 install ament_flake8 ament_pep257
        ament_flake8 src/
        ament_pep257 src/
        
    - name: Check C++ formatting
      run: |
        sudo apt install -y clang-format
        find src/ -name "*.cpp" -o -name "*.hpp" | xargs clang-format --dry-run -Werror
```

### Simulation Testing in CI

```yaml
name: Gazebo Simulation Test

on: [push, pull_request]

jobs:
  sim-test:
    runs-on: ubuntu-22.04
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup ROS2 and Gazebo
      run: |
        sudo apt update
        sudo apt install -y ros-humble-gazebo-ros-pkgs
        
    - name: Build workspace
      run: |
        source /opt/ros/humble/setup.bash
        colcon build
        
    - name: Run simulation test
      run: |
        source install/setup.bash
        timeout 60 ros2 launch my_robot_bringup gazebo_test.launch.py &
        sleep 10
        ros2 topic echo /scan --once
        ros2 topic echo /odom --once
        # If topics publish data, test passes
```

### Pre-commit Hooks (Run Tests Locally Before Push)

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
        
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
```

Install and use:

```bash
pip3 install pre-commit
pre-commit install

# Now every commit runs these checks automatically
git commit -m "feat: add new sensor driver"
# → Checks trailing whitespace, file endings, YAML syntax, file sizes, Python formatting
```

---

## Real-World Scenarios

### Scenario 1: Collaborative Parameter Tuning

**Problem:** Two engineers tune navigation parameters simultaneously.

```bash
# Engineer A
git checkout -b tune/increase-max-vel
# Modify nav2_params.yaml: max_vel_x: 0.5 → 0.8
git commit -m "perf(nav): increase max velocity for faster delivery"

# Engineer B (doesn't know about A's work)
git checkout -b tune/reduce-inflation
# Modify nav2_params.yaml: inflation_radius: 0.55 → 0.50
git commit -m "fix(nav): reduce inflation for narrow aisles"

# Engineer A merges first
git checkout develop
git merge tune/increase-max-vel  # ✅ Success

# Engineer B tries to merge
git merge tune/reduce-inflation  # ⚠️ CONFLICT!
```

**Resolve conflict:**

```bash
git status
# Both modified: config/nav2_params.yaml

# Open file, see conflict markers:
<<<<<<< HEAD
max_vel_x: 0.8
inflation_radius: 0.55
=======
max_vel_x: 0.5
inflation_radius: 0.50
>>>>>>> tune/reduce-inflation

# Manually merge:
max_vel_x: 0.8           # Keep A's change
inflation_radius: 0.50   # Keep B's change

# Complete merge
git add config/nav2_params.yaml
git commit -m "merge: combine velocity and inflation tuning"
```

### Scenario 2: Reverting a Broken Deploy

```bash
# You deploy commit abc123 to robot, it crashes
git log --oneline
abc123 feat: new path planner
def456 fix: costmap update
789abc refactor: cleanup

# Revert specific commit (creates new commit that undoes changes)
git revert abc123
git push origin main

# Robot now runs with old planner, but history preserved
```

### Scenario 3: Cherry-Picking a Fix Across Branches

```bash
# Bug fix committed to develop, but main needs it NOW
git checkout develop
git log --oneline
abc123 fix(safety): add timeout to sensor watchdog

# Apply just that commit to main
git checkout main
git cherry-pick abc123
git push origin main
```

### Scenario 4: Bisecting to Find When Bug Was Introduced

```bash
# Navigation worked 2 weeks ago, broken now. Which commit broke it?
git bisect start
git bisect bad                    # Current commit is broken
git bisect good v1.2.0            # v1.2.0 tag was working

# Git checks out middle commit
# Test if navigation works...
ros2 launch my_robot navigation.launch.py
# If broken:
git bisect bad
# If works:
git bisect good

# Repeat until Git finds the exact breaking commit
git bisect reset  # When done
```

---

## Quick Reference

### Daily Workflow

```bash
# Morning: Update your branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-work

# Work, commit atomically
git add specific_files
git commit -m "feat: descriptive message"

# Before lunch: Push backup
git push origin feature/my-work

# End of day: Ensure tests pass
colcon test
git push origin feature/my-work

# Create pull request on GitHub
# After review + approval → Merge to develop
```

### Emergency Commands

```bash
# Undo uncommitted changes
git checkout -- file.cpp

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Temporarily save changes (for branch switching)
git stash
git checkout other-branch
git stash pop

# See what changed
git diff
git diff HEAD~1  # Compare with previous commit

# Blame (who wrote this line?)
git blame src/controller.cpp
```

### Collaboration

```bash
# Update from remote
git fetch origin
git merge origin/develop

# Or in one step
git pull origin develop

# Push your branch
git push origin feature/my-work

# See all branches
git branch -a

# Delete merged branch
git branch -d feature/old-feature
```

---

## Interview Questions You Should Ace

**Q: "Explain the difference between `git merge` and `git rebase`."**

**A:** 
- **Merge:** Creates a merge commit, preserves complete history. Use for integrating feature branches.
- **Rebase:** Replays commits on top of another branch, creates linear history. Use for cleaning up before merging.

```bash
# Merge (creates merge commit)
git checkout develop
git merge feature/my-work
# Result: develop has a merge commit with two parents

# Rebase (rewrites history)
git checkout feature/my-work
git rebase develop
# Result: feature commits replayed on top of develop
```

**Q: "How would you set up CI/CD for a ROS2 navigation stack?"**

**A:** GitHub Actions workflow that:
1. Builds in Docker container with ROS2 Humble
2. Runs `colcon build` to check compilation
3. Runs `colcon test` for unit/integration tests
4. Launches Gazebo simulation, verifies topics publish
5. Runs navigation to goal, checks success rate
6. Only allows merge if all tests pass

**Q: "Your robot crashes in production. How do you investigate with Git?"**

**A:**
1. `git log --oneline` - check recent changes
2. `git diff v1.0.0 HEAD` - compare working version to current
3. `git bisect` - binary search for breaking commit
4. `git revert <bad_commit>` - deploy fix immediately
5. Create `hotfix/` branch to properly fix root cause

---

## Key Takeaways

1. **Atomic commits** = easier debugging, cleaner history
2. **Branch for every feature/fix** = parallel work without conflicts
3. **CI/CD catches bugs before hardware testing** = saves robot time
4. **Good commit messages = documentation** for 6-months-from-now you
5. **Git is not backup** = it's for collaboration and history

**Your Git workflow should be as reliable as your robot's sensors.**

---
