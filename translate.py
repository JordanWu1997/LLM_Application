#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
r"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2026-02-14 02:31:37             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import json
import math
import shutil
import sys

import requests
from langdetect import DetectorFactory, detect

# Consistent language detection
DetectorFactory.seed = 0

# Full list of supported languages
LANG_DATA = [("aa", "Afar"), ("ab", "Abkhazian"), ("af", "Afrikaans"),
             ("ak", "Akan"), ("am", "Amharic"), ("ar", "Arabic"),
             ("as", "Assamese"), ("az", "Azerbaijani"), ("be", "Belarusian"),
             ("bg", "Bulgarian"), ("bm", "Bambara"), ("bn", "Bengali"),
             ("bo", "Tibetan"), ("br", "Breton"), ("bs", "Bosnian"),
             ("ca", "Catalan"), ("ce", "Chechen"), ("co", "Corsican"),
             ("cs", "Czech"), ("cv", "Chuvash"), ("cy", "Welsh"),
             ("da", "Danish"), ("de", "German"), ("dv", "Divehi"),
             ("dz", "Dzongkha"), ("ee", "Ewe"), ("el", "Greek"),
             ("en", "English"), ("eo", "Esperanto"), ("es", "Spanish"),
             ("et", "Estonian"), ("eu", "Basque"), ("fa", "Persian"),
             ("ff", "Fulah"), ("fi", "Finnish"), ("fil", "Filipino"),
             ("fo", "Faroese"), ("fr", "French"), ("fy", "Western Frisian"),
             ("ga", "Irish"), ("gd", "Gaelic"), ("gl", "Galician"),
             ("gn", "Guarani"), ("gu", "Gujarati"), ("gv", "Manx"),
             ("ha", "Hausa"), ("he", "Hebrew"), ("hi", "Hindi"),
             ("hr", "Croatian"), ("ht", "Haitian"), ("hu", "Hungarian"),
             ("hy", "Armenian"), ("ia", "Interlingua"), ("id", "Indonesian"),
             ("ie", "Interlingue"), ("ig", "Igbo"), ("ii", "Sichuan Yi"),
             ("ik", "Inupiaq"), ("io", "Ido"), ("is", "Icelandic"),
             ("it", "Italian"), ("iu", "Inuktitut"), ("ja", "Japanese"),
             ("jv", "Javanes"), ("ka", "Georgian"), ("ki", "Kikuyu"),
             ("kk", "Kazakh"), ("kl", "Kalaallisut"), ("km", "Khmer"),
             ("kn", "Kannada"), ("ko", "Korean"), ("ks", "Kashmiri"),
             ("ku", "Kurdish"), ("kw", "Cornish"), ("ky", "Kyrgyz"),
             ("la", "Latin"), ("lb", "Luxembourgish"), ("lg", "Ganda"),
             ("ln", "Lingala"), ("lo", "Lao"), ("lt", "Lithuanian"),
             ("lu", "Luba-Katanga"), ("lv", "Latvian"), ("mg", "Malagasy"),
             ("mi", "Maori"), ("mk", "Macedonian"), ("ml", "Malayalam"),
             ("mn", "Mongolian"), ("mr", "Marathi"), ("ms", "Malay"),
             ("mt", "Maltese"), ("my", "Burmese"), ("nb", "Norwegian Bk"),
             ("nd", "North Ndebele"), ("ne", "Nepali"), ("nl", "Dutch"),
             ("nn", "Norwegian Ny"), ("no", "Norwegian"),
             ("nr", "South Ndebele"), ("nv", "Navajo"), ("ny", "Chichewa"),
             ("oc", "Occitan"), ("om", "Oromo"), ("or", "Oriya"),
             ("os", "Ossetian"), ("pa", "Punjabi"), ("pl", "Polish"),
             ("ps", "Pashto"), ("pt", "Portuguese"), ("qu", "Quechua"),
             ("rm", "Romansh"), ("rn", "Rundi"), ("ro", "Romanian"),
             ("ru", "Russian"), ("rw", "Kinyarwanda"), ("sa", "Sanskrit"),
             ("sc", "Sardinian"), ("sd", "Sindhi"), ("se", "Northern Sami"),
             ("sg", "Sango"), ("si", "Sinhala"), ("sk", "Slovak"),
             ("sl", "Slovenian"), ("sn", "Shona"), ("so", "Somali"),
             ("sq", "Albanian"), ("sr", "Serbian"), ("ss", "Swati"),
             ("st", "Southern Sotho"), ("su", "Sundanese"), ("sv", "Swedish"),
             ("sw", "Swahili"), ("ta", "Tamil"), ("te", "Telugu"),
             ("tg", "Tajik"), ("th", "Thai"), ("ti", "Tigrinya"),
             ("tk", "Turkmen"), ("tl", "Tagalog"), ("tn", "Tswana"),
             ("to", "Tonga"), ("tr", "Turkish"), ("ts", "Tsonga"),
             ("tt", "Tatar"), ("ug", "Uyghur"), ("uk", "Ukrainian"),
             ("ur", "Urdu"), ("uz", "Uzbek"), ("ve", "Venda"),
             ("vi", "Vietnamese"), ("vo", "Volapük"), ("wa", "Walloon"),
             ("wo", "Wolof"), ("xh", "Xhosa"), ("yi", "Yiddish"),
             ("yo", "Yoruba"), ("za", "Zhuang"), ("zh", "Chinese"),
             ("zu", "Zulu")]


def display_grid_menu(label, lang_data):
    item_width = 24
    search_query = ""
    current_page = 0
    while True:
        term_width, term_height = shutil.get_terminal_size()
        num_cols = max(1, (term_width - 2) // item_width)
        num_rows = max(1, term_height - 10)
        page_size = num_cols * num_rows

        filtered = [(c, n) for c, n in lang_data
                    if search_query in n.lower() or search_query in c.lower()]
        total_pages = math.ceil(len(filtered) / page_size) if filtered else 1
        if current_page >= total_pages: current_page = 0

        print("\033c", end="")
        print(f" SELECT {label.upper()} ".center(term_width, "="))
        print(
            f" Search: '{search_query} or '[None]' | Page {current_page+1}/{total_pages} "
            .center(term_width))
        print("-" * term_width)

        start = current_page * page_size
        page_items = filtered[start:start + page_size]
        for r in range(num_rows):
            row_cells = []
            for c in range(num_cols):
                idx = r + (c * num_rows)
                if idx < len(page_items):
                    code, name = page_items[idx]
                    orig_idx = lang_data.index((code, name)) + 1
                    cell = f"{str(orig_idx).rjust(3)}. {name[:12]} ({code})"
                    row_cells.append(cell.ljust(item_width))
            if any(cell.strip() for cell in row_cells):
                print(" " + "".join(row_cells))

        print("-" * term_width)
        print(" [N]ext | [P]rev | [S]earch | [R]eset | [EXIT] ".center(
            term_width))
        choice = input(f"\nChoice: ").strip().lower()
        if choice.isdigit():
            val = int(choice) - 1
            if 0 <= val < len(lang_data): return lang_data[val]
        elif choice == 'n': current_page = (current_page + 1) % total_pages
        elif choice == 'p': current_page = (current_page - 1) % total_pages
        elif choice == 's':
            search_query = input("Search Term: ").strip().lower()
            current_page = 0
        elif choice == 'r':
            search_query = ""
            current_page = 0
        elif choice == 'exit':
            sys.exit()


def get_ollama_response(text,
                        src,
                        tgt,
                        url,
                        model="translategemma:4b",
                        stream=True):
    prompt = (
        f"You are a professional {src[1]} ({src[0]}) to {tgt[1]} ({tgt[0]}) translator. "
        f"Convey meaning precisely. Produce only the {tgt[1]} translation. Translate:\n\n{text}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.2
        }
    }
    try:
        response = requests.post(f"{url}/api/generate",
                                 json=payload,
                                 stream=stream)
        response.raise_for_status()
        full_res = ""
        if stream:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk.get("response", "")
                    print(content, end="", flush=True)
                    full_res += content
                    if chunk.get("done"):
                        final_chunk = chunk

            # # Clipboard
            # pyperclip.copy(full_res)

            # Extract Metrics
            # Note: total_duration is in nanoseconds
            total_dur = final_chunk.get("total_duration", 0) / 1e9
            in_tokens = final_chunk.get("prompt_eval_count", 0)
            out_tokens = final_chunk.get("eval_count", 0)
            tps = out_tokens / total_dur if total_dur > 0 else 0

            # Print Stat Block
            print(f"\n\n" + "📊 PERFORMANCE STATS ".center(40, "-"))
            print(f"⏱  Time Elapsed:  {total_dur:.2f}s")
            print(f"📥 Input Tokens:  {in_tokens}")
            print(f"📤 Output Tokens: {out_tokens}")
            print(f"⚡ Throughput:    {tps:.2f} tokens/sec")
            # print(f"📋 Status:        Copied to clipboard!")
            print(f"📋 Status:        DONE!")
            print("-" * 40)
            return full_res
        else:
            return response.json().get("response", "").strip()
    except Exception as e:
        return f"\n[Error]: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--target",
                        help="Target language code (e.g., 'zh')")
    parser.add_argument("-s",
                        "--source",
                        help="Source language code (e.g., 'en')")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="11434")
    parser.add_argument("--model", default="translategemma:4b")
    parser.add_argument("text", nargs="*", help="Text to translate (CLI mode)")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    model = args.model

    # 1. CLI MODE (Clean Output)
    if args.target and args.text:
        input_text = " ".join(args.text)
        # Handle Source
        if args.source:
            s_code = args.source
            s_name = next((n for c, n in LANG_DATA if c == s_code), "Source")
        else:
            s_code = detect(input_text)
            s_name = next((n for c, n in LANG_DATA if c == s_code), "Detected")

        # Handle Target
        t_code = args.target
        t_name = next((n for c, n in LANG_DATA if c == t_code), "Target")

        # Output ONLY the translation
        print(
            get_ollama_response(input_text, (s_code, s_name), (t_code, t_name),
                                url,
                                model=model,
                                stream=False))

    # 2. INTERACTIVE MODE
    else:
        target_lang = None
        if args.target:
            target_lang = next(
                ((c, n) for c, n in LANG_DATA if c == args.target), None)

        while True:
            if not target_lang:
                target_lang = display_grid_menu("Target", LANG_DATA)

            print(
                f"\n[Target: {target_lang[1]}] Enter text (Double Enter to translate, 'MENU', 'EXIT'):"
            )
            lines = []
            while True:
                line = input()
                if line.upper() == "MENU":
                    target_lang = None
                    break
                if line.upper() == "EXIT": sys.exit()
                if line == "": break
                lines.append(line)

            if target_lang and lines:
                text = "\n".join(lines).strip()
                if not text: continue

                # Use provided source or detect
                if args.source:
                    s_code, s_name = args.source, next(
                        (n for c, n in LANG_DATA if c == args.source),
                        "Source")
                else:
                    try:
                        s_code = detect(text)
                        s_name = next((n for c, n in LANG_DATA if c == s_code),
                                      "Detected")
                    except:
                        s_code, s_name = "en", "English"

                print(f"\n[{s_name} -> {target_lang[1]}]: ", end="")
                get_ollama_response(text, (s_code, s_name),
                                    target_lang,
                                    url,
                                    model=model,
                                    stream=True)
                print("\n" + "=" * 40)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
