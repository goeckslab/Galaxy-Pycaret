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
