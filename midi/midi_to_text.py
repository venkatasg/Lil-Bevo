import mido
import sys
import itertools

def encode_message(msg: mido.Message):
    type_code = {'note_on': 'n', 'note_off': 'o'}.get(msg.type)
    if type_code is not None:
        if msg.time > 0:
            yield f't{msg.time}'
        yield f'c{msg.channel}{type_code}{msg.note}'

def midi_to_text(midi_path):
    mid = mido.MidiFile(midi_path)
    return ' '.join(
        itertools.chain.from_iterable(
            map(encode_message, mido.merge_tracks(mid.tracks))
        )
    )

if __name__ == "__main__":
    for input_path in sys.argv[1:]:
        print(midi_to_text(input_path))
