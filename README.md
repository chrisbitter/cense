# CENSE

This project uses Machine Learning to learn the Loop-Wire Game

## Git Conventions

###Branch Names

Convention: **topic_short/initials/branchname**

####Topics

| Topic | topic_short | Description |
|---|---|---|
| ro | Robot | direct interfaces with the hardware |
| ag | Agent | learning algorithms |
| en | Environment | - |
| si | Simulation | - |
| mi | Middleware | managing settings and training |
| ui | User Interface | - |
| vr | Virtual Reality | - |

####Initials

WM: First letter of first and last name
- Max Muster &rarr; mm

HiWi: First two letters of first and last name
- Max Muster &rarr; mamu

####branch name

## Requirements

- Python 3.5

## Setup

- install wheels in Resources/python_wheels
  - in Console: pip install path_to_wheel

## Hacks

- in VideoCapture/__init__.py, change im = Image.fromstring to im = Image.frombytes
  - there will be an error when running pointing to the right line of code.
- in keras/utils/generic_utils change line 175 to "code = marshal.dumps(func.__code__).replace(b'\\',b'/').decode('raw_unicode_escape')"
  - needed because of lam
## Remote Access

- copy Resources/credentials_template.json file
- rename to credentials.json
- enter valid credentials
- **YOU OBVIOUSLY MUST NOT DISTRIBUTE credentials.json to Git!**
  - Note: credentials.json is in .gitignore