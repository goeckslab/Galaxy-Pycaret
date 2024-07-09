# Galaxy-Pycaret
A library of Galaxy machine learning tools based on PyCaret — part of the Galaxy ML2 tools, aiming to provide simple, powerful, and robust machine learning capabilities for Galaxy users.

# Install Galaxy-Pycaret into Galaxy

* Update `tool_conf.xml` to include Galaxy-Pycaret tools. See [documentation](https://docs.galaxyproject.org/en/master/admin/tool_panel.html) for more details. This is an example:
```
<section id="pycaret" name="Pycaret Applications">
  <tool file="galaxy-pycaret/tools/pycaret_train.xml" />
</section>
```

* Configure the `job_conf.yml` under `lib/galaxy/config/sample` to enable the docker for the environment you want the Ludwig related job running in. This is an example:
```
execution:
 default: local
 environments:
   local:
     runner: local
     docker_enabled: true
```
If you are using an older version of Galaxy, then `job_conf.xml` would be something you want to configure instead of `job_conf.yml`. Then you would want to configure destination instead of execution and environment. 
See [documentation](https://docs.galaxyproject.org/en/master/admin/jobs.html#running-jobs-in-containers) for job_conf configuration. 
* If you haven’t set `sanitize_all_html: false` in `galaxy.yml`, please set it to False to enable our HTML report functionality.
* Should be good to go. 

# Make contributions

## Getting Started

To get started, you’ll need to fork the repository, clone it locally, and create a new branch for your contributions.

1. **Fork the Repository**: Click the "Fork" button at the top right of this page.
2. **Clone the Fork**:
  ```bash
    git clone https://github.com/<your-username>/Galaxy-Pycaret.git
    cd <your-repo>
  ```
3. **Create a Feature/hotfix/bugfix Branch**:
  ```bash
    git checkout -b feature/<feature-branch-name>
  ```
  or
  ```bash
    git checkout -b hotfix/<hoxfix-branch-name>
  ```
  or
  ```bash
    git checkout -b bugfix/<bugfix-branch-name>
  ```

## How We Manage the Repo

We follow a structured branching and merging strategy to ensure code quality and stability.

1. **Main Branches**:
   - **`main`**: Contains production-ready code.
   - **`dev`**: Contains code that is ready for the next release.

2. **Supporting Branches**:
   - **Feature Branches**: Created from `dev` for new features.
   - **Release Branches**: Created from `dev` when preparing a new release.
   - **Hotfix Branches**: Created from `main` for critical fixes in production. 

### Workflow

- **Feature Development**: 
  - Branch from `dev`.
  - Work on your feature.
  - Submit a Pull Request (PR) to `dev`.
- **Hotfixes**: 
  - Branch from `main`.
  - Fix the issue.
  - Merge back into both `main` and `dev`.

## Contribution Guidelines

We welcome contributions of all kinds. To make contributions easy and effective, please follow these guidelines:

1. **Create an Issue**: Before starting work on a major change, create an issue to discuss it.
2. **Fork and Branch**: Fork the repo and create a feature branch.
3. **Write Tests**: Ensure your changes are well-tested if applicable.
4. **Code Style**: Follow the project’s coding conventions.
5. **Commit Messages**: Write clear and concise commit messages.
6. **Pull Request**: Submit a PR to the `dev` branch. Ensure your PR description is clear and includes the issue number.

### Submitting a Pull Request

1. **Push your Branch**:
    ```bash
    git push origin feature/<feature-branch-name>
    ```
2. **Open a Pull Request**:
   - Navigate to the original repository where you created your fork.
   - Click on the "New Pull Request" button.
   - Select `dev` as the base branch and your feature branch as the compare branch. 
   - Fill in the PR template with details about your changes.

3. **Rebase or Merge `dev` into Your Feature Branch**:
    - Before submitting your PR or when `dev` has been updated, rebase or merge `dev` into your feature branch to ensure your branch is up to date:
    
4. **Resolve Conflicts**:
    - If there are any conflicts during the rebase or merge, Git will pause and allow you to resolve the conflicts.

5. **Review Process**: Your PR will be reviewed by a team member. Please address any feedback and update your PR as needed.