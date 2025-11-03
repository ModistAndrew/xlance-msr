COARSE_LEVEL_INSTRUMENTS = {
    "vocals",
    "bass",
    "drums",
    "guitar",
    "other_plucked",
    "percussion",
    "piano",
    "other_keys",
    "bowed_strings",
    "wind",
    "other",
}

TAXONOMY = {
    "vocals": [
        "lead_male_singer",
        "lead_female_singer", 
        "human_choir",
        "background_vocals",
        "other_vocoder_beatboxing_etc",
    ],
    "bass": [
        "bass_guitar",
        "bass_synthesizer_moog_etc",
        "contrabass_double_bass_bass_of_instrings",
        "tuba_bass_of_brass", 
        "bassoon_bass_of_woodwind",
    ],
    "drums": [
        "snare_drum",
        "toms",
        "kick_drum",
        "cymbals",
        "overheads",
        "full_acoustic_drumkit",
        "drum_machine",
        "hi_hat"
    ],
    "other": [
        "fx_processed_sound_scratches_gun_shots_explosions_etc",
        "click_track",
    ],
    "guitar": [
        "clean_electric_guitar",
        "distorted_electric_guitar",
        "lap_steel_guitar_or_slide_guitar",
        "acoustic_guitar",
    ],
    "other_plucked": ["banjo_mandolin_ukulele_harp_etc"],
    "percussion": [
        "a_tonal_percussion_claps_shakers_congas_cowbell_etc",
        "pitched_percussion_mallets_glockenspiel_",
    ],
    "piano": [
        "grand_piano", 
        "electric_piano_rhodes_wurlitzer_piano_sound_alike",
    ],
    "other_keys": [
        "organ_electric_organ",
        "synth_pad",
        "synth_lead",
        "other_sounds_hapischord_melotron_etc",
    ],
    "bowed_strings": [
        "violin_solo",
        "viola_solo",
        "cello_solo", 
        "violin_section",
        "viola_section",
        "cello_section",
        "string_section",
        "other_strings",
    ],
    "wind": [
        "brass_trumpet_trombone_french_horn_brass_etc",
        "flutes_piccolo_bamboo_flute_panpipes_flutes_etc",
        "reeds_saxophone_clarinets_oboe_english_horn_bagpipe",
        "other_wind",
    ],
}

TARGET_STEM_MAPPING = {
    "vox": [
        ("vocals", "lead_male_singer"), 
        ("vocals", "lead_female_singer"),
        ("vocals", "human_choir"),
        ("vocals", "background_vocals")
    ],
    "bass": [
        ("bass", "bass_guitar"),
        ("bass", "bass_synthesizer_moog_etc")
    ],
    "drums": [
        ("drums", "snare_drum"), 
        ("drums", "toms"), 
        ("drums", "kick_drum"), 
        ("drums", "cymbals"), 
        ("drums", "overheads"), 
        ("drums", "full_acoustic_drumkit"),
        ("drums", "hi_hat")
    ],
    "gtr": [
        ("guitar", "clean_electric_guitar"), 
        ("guitar", "distorted_electric_guitar"), 
        ("guitar", "lap_steel_guitar_or_slide_guitar"), 
        ("guitar", "acoustic_guitar")
    ],
    "key": [
        ("piano", "grand_piano"), 
        ("piano", "electric_piano_rhodes_wurlitzer_piano_sound_alike")
    ],
    "perc": [
        ("percussion", "a_tonal_percussion_claps_shakers_congas_cowbell_etc"), 
        ("percussion", "pitched_percussion_mallets_glockenspiel_")
    ],
    "syn": [
        ("other_keys", "synth_pad"), 
        ("other_keys", "synth_lead")
    ],
    "orch": [
        ("bass", "contrabass_double_bass_bass_of_instrings"),
        ("bass", "tuba_bass_of_brass"),
        ("bass", "bassoon_bass_of_woodwind"),
        ("bowed_strings", "violin_solo"),
        ("bowed_strings", "viola_solo"),
        ("bowed_strings", "cello_solo"), 
        ("bowed_strings", "violin_section"),
        ("bowed_strings", "viola_section"),
        ("bowed_strings", "cello_section"),
        ("bowed_strings", "string_section"),
        ("wind", "brass_trumpet_trombone_french_horn_brass_etc"),
        ("wind", "flutes_piccolo_bamboo_flute_panpipes_flutes_etc"),
        ("wind", "reeds_saxophone_clarinets_oboe_english_horn_bagpipe")
    ],
    "other": [
        ("vocals", "other_vocoder_beatboxing_etc"),
        ("drums", "drum_machine"),
        ("other", "fx_processed_sound_scratches_gun_shots_explosions_etc"),
        ("other", "click_track"),
        ("other_plucked", "banjo_mandolin_ukulele_harp_etc"),
        ("other_keys", "organ_electric_organ"),
        ("other_keys", "other_sounds_hapischord_melotron_etc"),
        ("bowed_strings", "other_strings"),
        ("wind", "other_wind")
    ]
}

def get_target_stem_pairs(target_stem):
    return TARGET_STEM_MAPPING[target_stem]

def get_banned_other_pairs(target_stem):
    return TARGET_STEM_MAPPING["other"]