from datasets import load_dataset
from random import choice, seed

def main():
    seed(78727)
    
    dataset = load_dataset('text', data_files='babylm_data/maestro/midi.txt')
    
    all_music_notes = ' '.join([x['text'] for x in dataset['train']])
    
    unique_notes = list(set(all_music_notes.split()))
    num_notes = len(all_music_notes.split())
    import ipdb;ipdb.set_trace()
    # Generate random notes of same length and write to file random_midi
    random_notes = [choice(unique_notes) for _ in range(num_notes)]
    random_notes = '\n'.join([' '.join(random_notes[i:i+1000]) for i in range(0, num_notes, 1000)])
    
    
    with open('babylm_data/maestro/random_midi.txt', 'w') as f:
        f.write(random_notes)

if __name__=="__main__":
    main()
