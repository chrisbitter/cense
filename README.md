### README.MD ###

# CENSE

This project uses Machine Learning to learn the Loop-Wire Game


## Requirements

- Python 3.5

## Setup

- install wheels in Resources/python_wheels
  - in Console: pip install path_to_wheel

## Hacks

- in VideoCapture/__init__.py, change im = Image.fromstring to im = Image.frombytes
  - there will be an error when running pointing to the right line of code.

## Remote Access

- copy Resources/credentials_template.json file
- rename to credentials.json
- enter valid credentials
- **YOU OBVIOUSLY MUST NOT DISTRIBUTE credentials.json to Git!**
  - Note: credentials.json is in .gitignore


Programm
    VorStart
        tmp≔p[0,0,0,0,0,0]
        new_pose≔Werkzeug
        goal_reached≔ True
        pose_received≔ False
        abort≔ False
        sync()
Roboterprogramm
    If pose_received
        If  not abort
            FahreLinear
                new_pose
        Else
            abort≔ False
        pose_received≔ False
        goal_reached≔ True
    sync()
Thread_1
    If goal_reached
        tmp≔p[0,0,0,0,0,0]
        tmp[0] = read_input_float_register(0)
        tmp[1] = read_input_float_register(1)
        tmp[2] = read_input_float_register(2)
        tmp[3] = read_input_float_register(3)
        tmp[4] = read_input_float_register(4)
        tmp[5] = read_input_float_register(5)
    If (tmp ≠ new_pose) and (tmp≠p[0,0,0,0,0,0])
        new_pose≔tmp
        pose_received≔ True
        goal_reached≔ False
    If read_input_integer_register(0)
        abort≔ True
    sync()
