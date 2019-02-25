
# Myo armband

The Myo band was created by Thalmic Labs. This device contains 9 dimensions
of data called inertial measurement units (IMU). These are represented
by electronic sensors that return information about speed, orientation and gravity.
The device also provides 8 electrodes that allow to return information
about the current flowing through the muscles - electromyography.

# Myo and Polish Sign Language

Difficult problem is to facilitate the communication of a deaf
person who communicates using sign language with a person who does
not know such a way of conversation. THis work try
to analyze hands movement in real time into written or spoken language.
To achieve this, the experiment uses machine learning, which is
used to recognize words in hand movements.

The project classified 18 words and with the proposed approach satisfactory
results were obtained showing performance of 91%.

# How to

```
source venv/bin/activate --ppython 3.6.5
source venv/bin/activate
pip install -r requirements.txt
```

Run bellow command to connect to already
running Myo command and send by OSC protocole data from thin armband
```
python run_recorder.py --myo --address 4000
```

Run bellow command to listen on OSC server that send data from myo
```
python run_recorder.py --recording --address 4000

```

# Run tests

```
pytest
```