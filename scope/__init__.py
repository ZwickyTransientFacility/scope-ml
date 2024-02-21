import shutil
import os
import site

# Below code adapted from https://github.com/skyportal/skyportal/blob/main/skyportal/__init__.py
# 2022-10-18
__version__ = "0.9.0"

if "dev" in __version__:
    # Append last commit date and hash to dev version information, if available

    import subprocess
    import os.path

    try:
        p = subprocess.Popen(
            ["git", "log", "-1", '--format="%h %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode("utf-8")
                .strip()
                .replace('"', "")
                .split("T")[0]
                .replace("-", "")
                .split()
            )

            __version__ = "+".join(
                [tag for tag in __version__.split("+") if not tag.startswith("git")]
            )
            __version__ += f"+git{git_date}.{git_hash}"


def initialize():
    """create directories, copy config and data files"""
    main_dir = "scope"
    scope_dirs = ["tools"]
    os.makedirs(main_dir, exist_ok=True)
    for directory in scope_dirs:
        os.makedirs(f"{main_dir}/{directory}", exist_ok=True)

    site_packages_path = site.getsitepackages()[0]
    default_config_name = "config.defaults.yaml"
    copied_config_name = "config.yaml"
    tools_dir = "tools"
    mappers = [
        "golden_dataset_mapper.json",
        "fritz_mapper.json",
        "DNN_AL_mapper.json",
        "XGB_AL_mapper.json",
        "local_scope_ztfid.csv",
        "local_scope_radec.csv",
    ]

    print()
    # Copy config defaults to new directory strucutre if needed
    if not os.path.exists(f"{main_dir}/{copied_config_name}"):
        shutil.copy(
            f"{site_packages_path}/{default_config_name}",
            f"{main_dir}/{default_config_name}",
        )
        shutil.copy(
            f"{site_packages_path}/{default_config_name}",
            f"{main_dir}/{copied_config_name}",
        )
        print(
            f"Created new '{copied_config_name}' config file. Please customize/add tokens there before running scope."
        )
    else:
        print(
            f"Warning: {copied_config_name} already exists in the '{main_dir}' directory."
        )

    print()
    for mapper in mappers:
        print(f"Copying default data '{mapper}' to '{main_dir}/{tools_dir}'")
        shutil.copy(
            f"{site_packages_path}/{tools_dir}/{mapper}",
            f"{main_dir}/{tools_dir}/{mapper}",
        )

    print()
    print(f"scope-ml initialized. Run scripts from '{main_dir}' directory.")
