#!/usr/bin/env python
from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger

try:
    unicode
except:
    unicode = str


## ref https://spacy.io/docs/usage/entity-recognition#updating
# Ranjeet - 18-Sep
def train_ner(nlp, train_data, entity_types):
    # Add new words to vocab.
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]

    # Train NER.
    ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            ner.update(doc, gold)
    return ner


def save_model(ner, model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, unicode):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


def main(model_dir=None):
    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)

    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    train_data = [
        ('Samsung India electronics Pvt. Ltd 20th to 24th '
         'floor,two horizon center,golf course road,sector-43,dlf phase v '
         'Gurgaon haryana 122002,india', [(0, 34, 'COMPANY')]),  ## Index pos (0-34)
        ('I like London and Berlin.', [(7, 13, 'LOC'), (18, 24, 'LOC')])
    ]
    ner = train_ner(nlp, train_data, ['COMPANY', 'LOC'])

    # Example
    doc = nlp.make_doc(
        'Nokia India electronics Pvt. Ltd 20th to 24th floor,two horizon center,golf course road,sector-43,dlf phase v Gurgaon haryana 122002,india')
    nlp.tagger(doc)
    ner(doc)
    for word in doc:
        # show only company tag.. rest will be address
        if word.ent_type_ == 'COMPANY':
            print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)

    if model_dir is not None:
        save_model(ner, model_dir)


if __name__ == '__main__':
    main('ner')

