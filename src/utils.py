from aiolimiter import AsyncLimiter
from tqdm.auto import tqdm
from time import perf_counter
from collections import deque
import inspect
import asyncio
from itertools import chain
import re


class SectionNode:
    def __init__(self, level=0, name="", text=""):
        self.level = level
        self.name = name
        self.text = text
        self.parent = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        node._set_parent(self)

    def insert_child(self, node):
        if node.level > self.level:
            if node.level - 1 == self.level or not self.children:
                self.add_child(node)
            else:
                self.children[-1].insert_child(node)

    def _set_parent(self, node):
        self.parent = node

    def __repr__(self):
        return self.compile_text()

    def compile_text(self, max_depth=-1, new_level=-1):
        max_depth -= 1
        level = self.level if new_level < 0 else new_level
        title = "#"*level + f" {self.name}"
        return "\n".join(
            [title, self.text, *(
                ch.compile_text(max_depth, new_level=new_level+1) for ch in self.children if max_depth != 0
            )])

    def trim_empty_leaves(self):
        retain = []
        for ch in self.children:
            if (ch.children or ch.text):
                retain.append(ch)
                ch.trim_empty_leaves()
            # else:
            #     print(ch)
        self.children = retain


empty_node = SectionNode()


def match_node(query, node_list):
    return next(
        chain(filter(
            lambda x: re.match(query.lower(), x.name.lower()),
            node_list
        ),
            [empty_node])
    )


code_section_pattern = "(```.{8,}?```)"
code_section_pattern = re.compile(code_section_pattern, flags=re.M | re.DOTALL | re.U)


def find_code_sections(text):
    return list(code_section_pattern.findall(text))


def mask_substrs(text, substrs):
    mask_dict = {}
    for s, sub in enumerate(substrs):
        mask = f"[SUB{s}]"
        mask = f"{mask}{'?'*(len(sub)-2*len(mask))}{mask}"
        assert len(mask) == len(sub)
        mask_dict[mask] = sub
        text = text.replace(sub, mask)
    return text, mask_dict


def unmask_text(text, mask_dict):
    for mask, sub in mask_dict.items():
        text = text.replace(mask, sub)
    return text


HTML_MATH_PATTERN = r"\[math\]\\displaystyle{[\n\s]*(.+?)[\n\s]*}\[/math\]"
HTML_MATH_PATTERN = re.compile(HTML_MATH_PATTERN, flags=re.M | re.U | re.DOTALL)


def html_to_latex(latex_text):
    res = HTML_MATH_PATTERN.sub(r"$$\1$$", latex_text)
    res = res.replace(" $$", " $").replace("$$ ", "$ ")
    return res


def extract_title(html):
    return html.find_all("h1", attrs={"id": "firstHeading"})[0].text


markdown_link_pattern = "(!?\[.*\]\(.+?\))"
markdown_link_pattern = re.compile(markdown_link_pattern)


def remove_markdown_links(text):
    return markdown_link_pattern.sub("", text)


def takefirst(arr):
    return next(
        chain(arr, [None])
    )


class catchtime:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'


class AsyncList:
    def __init__(self):
        self.contents = []

    def append(self, item):
        self.contents.append(item)

    @property
    def couroutine_ids(self):
        return [i for i, item in enumerate(self.contents) if inspect.iscoroutine(item)]

    async def complete_couroutines(self, batch_size=10, verbose=False, max_couroutines=None):
        if max_couroutines is None:
            max_couroutines = batch_size * 2

        if not self.couroutine_ids:
            return
        lim = AsyncLimiter(batch_size, time_period=1.0)
        sem = asyncio.Semaphore(max_couroutines)

        async def batch_run(idx):
            async with sem:
                async with lim:
                    task = self.contents[idx]
                    res = await task
                    self.contents[idx] = res

        tasks = [asyncio.create_task(batch_run(i)) for i in self.couroutine_ids]
        try:
            await tqdm.gather(*tasks, disable=not verbose, leave=False, position=1)
        except BaseException as e:
            for t in tasks:
                if not t.done() and not t.cancelled():
                    t.cancel()
            raise e

    def __getitem__(self, key):
        return self.contents[key]

    def __repr__(self):
        return repr(self.contents)

    def __len__(self):
        return len(self.contents)

    async def to_list(self, **kwargs):
        await self.complete_couroutines(**kwargs)
        return self.contents


class AsyncDict:
    def __init__(self):
        self.contents = {}

    def __setitem__(self, key, value):
        self.contents[key] = value

    def __getitem__(self, key):
        return self.contents[key]

    def __len__(self):
        return len(self.contents)

    @property
    def couroutine_keys(self):
        return [k for k, v in self.contents.items() if inspect.iscoroutine(v)]

    async def complete_couroutines(self, batch_size=10, verbose=False, max_couroutines=None):
        if max_couroutines is None:
            max_couroutines = batch_size * 2

        if not self.couroutine_keys:
            return
        lim = AsyncLimiter(batch_size, time_period=1.0)
        sem = asyncio.Semaphore(max_couroutines)

        async def batch_run(key):
            async with sem:
                async with lim:
                    task = self.contents[key]
                    res = await task
                    self.contents[key] = res

        tasks = [asyncio.create_task(batch_run(k)) for k in self.couroutine_keys]
        try:
            await tqdm.gather(*tasks, disable=not verbose, leave=False, position=1)
        except BaseException as e:
            for t in tasks:
                if not t.done() and not t.cancelled():
                    t.cancel()
            raise e

    async def to_dict(self, **kwargs):
        await self.complete_couroutines(**kwargs)
        return dict(self.contents)


def persistent_coro(fun):
    async def wrapper(*args, **kwargs):
        try:
            return await fun(*args, **kwargs)
        except Exception as e:
            print('Coroutine failed', e)
            return wrapper(*args, **kwargs)
    return wrapper
