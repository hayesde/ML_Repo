# learning OAS specification with cpu and os example
This weeks lab you will be learning how to propelry use Open API specification or (OAS) from here on out. Following this specification will allow us to create a directory hierarchy of python files or modules that we can reference within a yaml file. This makes our applications super clean and readily transferable.

## Read about OAS

You are required to read about writing OAS definitions and the required reading can be found here:

[Defining OAS 3.0](https://swagger.io/docs/specification/basic-structure/)

[Defining OAS 2.0](https://swagger.io/docs/specification/2-0/basic-structure/)

This documentation walks you through the basics of your YAML file, we will be using YAML exculsively in this class, however if you want to use JSON for your project feel free.

# connexion

To conform to OAS standards we will be using connexion to handle the HTTP requests. The documentation is found here:

[connexion docs](https://connexion.readthedocs.io/en/latest/)

# goal

The outcome of this lab is for you to begin to understand how to write your own yaml files. I have provided for you a nice start.

# start

`make docker-all`

now remember in order to stop the service use `make docker-stop` in a seperate terminal. This should work seamless.

Look in os_pack there are two files. Looks at them. Redfine the operationid in the yaml file to point to the cpu_2020.py file. I have already created the def for the OS portion so the lab activity will be easy.

# what you need to do for the lab activity

create a new endpoint that tells the user what OS they are running by defining a path and operationid in the yaml file.

# homework

I will post on piazza but the links in this document must be read.
