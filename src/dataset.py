from .utils import persistent_coro
from .programs import AlgoWikiCleanData, AlgoWikiImplToCode
import json
from .utils import AsyncList
from .programs import AlgoWikiNormalizeGT
from .settings import ModuleSettings
import dspy
from .scraper import load_html
from pathlib import Path
from .utils import takefirst, extract_title, match_node, remove_markdown_links
from bs4 import BeautifulSoup
from html_to_markdown import convert
from dataclasses import asdict
from dataclasses import dataclass
from itertools import chain
from .utils import SectionNode, html_to_latex, find_code_sections, mask_substrs, unmask_text

import re

markdown_link_pattern = re.compile("\[(.+?)\]\(.+?\)")
markdown_cite_ref_pattern = re.compile("\[\[[0-9]+\]\]\(.+?\)")
markdown_image_pattern = re.compile("\[\!\[\].+?\]\(.+\)")


def remove_md_links(text):
    text = markdown_image_pattern.sub("", text)
    text = markdown_cite_ref_pattern.sub("", text)
    text = markdown_link_pattern.sub(r"\1", text)
    return text


@dataclass
class AlgoTable:
    complexity: str = "-1"
    inp_size: str = "-1"
    out_size: str = "-1"
    form_height: str = "-1"
    form_width: str = "-1"


algotable_name_map = {
    'Последовательная сложность': "complexity",
    'Объём входных данных': "inp_size",
    'Объём выходных данных': "out_size",
    'Высота ярусно-параллельной формы': "form_height",
    'Ширина ярусно-параллельной формы': "form_width",
}


def extract_algo_properties(algo_properties_table):
    prop_dict = {}

    tmp = algo_properties_table.find_all(
        'td', attrs={"style": "background-color:#eef; padding: 0.2em; font-weight: bold;"})
    for prop in tmp:
        prop_name = prop.text.strip()
        for s in prop.next_siblings:
            if s.text.strip() and s.span:
                prop_value = html_to_latex(s.span.text)
                prop_dict[prop_name] = prop_value
                break
    arg_dict = {algotable_name_map[k]: v for k, v in prop_dict.items() if k in algotable_name_map}
    res = AlgoTable(**arg_dict)
    return res


section_pattern = "(#+ )(.+?)\n+"  # \n"
section_pattern = re.compile(section_pattern, flags=re.M | re.DOTALL | re.U)


def extract_sections(markdown):
    code_strs = find_code_sections(markdown)
    markdown, mask_dict = mask_substrs(markdown, code_strs)
    tmp = list(section_pattern.finditer(markdown))

    levels, names = list(zip(*((match.group(1), match.group(2)) for match in tmp)))
    borders = list(chain(*((match.start(), match.end()) for match in tmp)))[1:] + [None]
    borders = [slice(l, r) for l, r in zip(borders[::2], borders[1::2])]
    markdown = unmask_text(markdown, mask_dict)

    sections = [(len(l.strip()), n, markdown[b]) for l, n, b in zip(levels, names, borders)]
    return sections


def build_section_tree(sections, trim_empty=True):
    tree = SectionNode(name="ROOT")

    node_list = [tree]
    for level, name, text in sections:
        node = SectionNode(level, name, text)
        # print(name, level)
        tree.insert_child(node)
        node_list.append(node)
    tree.trim_empty_leaves()
    return node_list


def load_external_code(file_name, code_dir="./data/external_code"):
    path = Path(code_dir) / f"{file_name}.code"
    code = None
    if path.exists():
        print(f"Found implementation for {path}")
        with open(path) as f:
            code = f.read()
            code = f'```\n{code}\n```'
    return code


def parse_algowiki_page(page, file_name):
    html = BeautifulSoup(page, features="html.parser")
    title = extract_title(html)
    algo_properties_table = takefirst((cand for cand in html.find_all("table") if 'Объём выходных данных' in cand.text))
    markdown = html_to_latex(convert(page).content)

    if algo_properties_table is None:
        return title, None, None, markdown
    target = extract_algo_properties(algo_properties_table)

    # print(markdown)
    sections = extract_sections(markdown)
    section_tree = build_section_tree(sections)

    impl_name = "программная реализация"
    impl_text = match_node(impl_name, section_tree)
    if not impl_text:
        # print("!!!", title)
        impl_text = None
    else:
        impl_text = impl_text.compile_text(new_level=1)
        if '```' not in impl_text:
            # if code available just replace all text with exact implementation
            impl_text = load_external_code(file_name)
        if impl_text:
            impl_text = remove_md_links(impl_text)

    math_desc_names = ["Общее описание алгоритма", "Математическое описание алгоритма",
                       "Вычислительное ядро алгоритма", "Макроструктура алгоритма"]
    math_desc_text = "\n".join(
        [match_node(name, section_tree).compile_text(1, new_level=1) for name in math_desc_names]
    )

    math_desc_text = remove_md_links(math_desc_text)

    input_data = {
        "math": math_desc_text,
        "code": impl_text
    }
    return title, target, input_data, markdown


@dataclass
class AlgoWikiRow:
    title: str
    math_desc: str
    code_desc: str
    python_desc: str
    markdown: str
    normalized_target: AlgoTable = None
    o_notation_target: AlgoTable = None
    html_file: str = None


async def normalize_table(table: AlgoTable):
    all_feats_raw = [table.complexity, table.inp_size, table.out_size, table.form_height, table.form_width]
    all_feats = "\n".join(f"- {feat}" for feat in all_feats_raw)
    program = AlgoWikiNormalizeGT()
    res = await program.acall(all_feats)

    normalized = [r.removeprefix("- ") for r in res.normalized.split("\n")]
    o_notation = [r.removeprefix("- ") for r in res.o_notation.split("\n")]
    assert len(normalized) == len(all_feats_raw)
    assert len(o_notation) == len(normalized)

    normalized = AlgoTable(*normalized)
    o_notation = AlgoTable(*o_notation)
    return normalized, o_notation


def dump_row(row, path):
    with open(path, encoding='utf-8', mode='w') as f:
        row = asdict(row)
        json.dump(row, f, ensure_ascii=False)


def load_row(path):
    with open(path, encoding='utf-8') as f:
        row = json.load(f)
        row = AlgoWikiRow(**row)
        return row


redacted_brackets_pattern = re.compile("\(.*?\[REDACTED\].*?\)")


def remove_redacted_fragments(text):
    text = redacted_brackets_pattern.sub("", text)
    text = text.replace("[REDACTED]", "")
    return text


async def clean_math_input(input_data, target):
    desc = input_data['math']
    all_feats = [target.complexity, target.inp_size, target.out_size, target.form_height, target.form_width]
    target = "\n".join(f"- {feat}" for feat in all_feats)
    program = AlgoWikiCleanData()
    res = await program.acall(desc, target)
    res = remove_redacted_fragments(res.cleaned)
    input_data['math'] = res


async def pythonize_code_input(input_data):
    desc = input_data['code']
    input_data['python'] = desc
    if desc:
        program = AlgoWikiImplToCode()
        res = await program.acall(desc)
        res = res.translation
        input_data['python'] = res


async def run_builder_agents(input_data, target, title, markdown, file_name, out_path):
    await clean_math_input(input_data, target)
    await pythonize_code_input(input_data)
    try:
        normalized, o_notation = await normalize_table(target)
    except Exception as e:
        print(e)
        return

    row = AlgoWikiRow(
        title,
        input_data['math'],
        input_data['code'],
        input_data['python'],
        markdown,
        normalized_target=normalized,
        o_notation_target=o_notation,
        html_file=file_name
    )

    dump_row(row, out_path)


def build_algowiki_example(source_path, out_dir: Path, force=False):
    page = load_html(source_path)
    file_name = source_path.stem
    out_path = out_dir / f"{file_name}.json"
    if out_path.exists() and not force:
        row = load_row(out_path)
        # check if code_desc can be filled
        if row.code_desc or not load_external_code(file_name):
            return
    title, target, input_data, markdown = parse_algowiki_page(page, file_name)
    if not target or not input_data or not any(input_data.values()):
        print(source_path)
        return
    return run_builder_agents(input_data, target, title, markdown, file_name, out_path)


async def build_rows(
    config: ModuleSettings, llm_name: str,
    source_dir: str = "./data/html/",
    out_dir: str = "./data/prepared/",
    force=False
):
    data_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    lm = dspy.LM(f"openai/{llm_name}", api_key=config.openai_server.key,
                 api_base=config.openai_server.url, model_type="chat", rollout_id=221)
    dspy.configure(lm=lm, **config.generation.model_dump())

    tasks = AsyncList()
    for fp in data_dir.glob("*.html"):
        task = build_algowiki_example(fp, out_dir, force=force)
        tasks.append(task)

    await tasks.complete_couroutines(batch_size=5, verbose=True)

    return


def collect_rows(
    path="./data/prepared/"
):
    data_dir = Path(path)

    examples = []
    for fp in data_dir.glob("*.json"):
        row = load_row(fp)
        examples.append(row)
    return examples


def algowiki_to_dict(row, message_field='python_desc', target_type='o_notation'):
    row = asdict(row)
    row['message'] = row.pop(message_field)
    row |= row.pop(f"{target_type}_target")
    return row


def build_dataset(full_only=True, limit=None, message_field='python_desc'):
    examples = collect_rows()
    dataset = [
        dspy.Example(algowiki_to_dict(ex, message_field=message_field)).with_inputs("message")
        for ex in examples if (ex.code_desc or not full_only)
    ][:limit]
    return dataset
