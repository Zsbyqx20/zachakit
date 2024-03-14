import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Literal

import tiktoken
from bidict import bidict
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from tiktoken import Encoding

from ..utils import estimate_tokens, load_azure_client, load_local_client, rjl, wjl
from . import MODEL_ALIAS, SUPPORTED_MODEL_NAMES, TOKEN_PRICE


class Query(BaseModel):
    name: str = "gpt-3.5-turbo-1106"
    message_param: List[ChatCompletionMessageParam]
    temperature: float = 0.5
    debug: bool = False
    run_limit: int = 10
    force_fail: bool = False

    def query(self, client: OpenAI, enc: Encoding = None):
        message: ChatCompletionMessage = ChatCompletionMessage(content="debug.", role="assistant")
        usage: CompletionUsage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        success = None
        if self.force_fail:
            message.content = "Fail."
            success = False
        elif not self.debug:
            message, usage, success = self._query(client)
        else:
            assert enc, "You need to specify a valid Encoding object for estimation of tokens."
            usage.total_tokens = usage.prompt_tokens = estimate_tokens(self.message_param, enc)
            success = True
        return message, usage, success

    def _query(self, client: OpenAI):
        run_times = 0
        success = None
        while run_times < self.run_limit:
            try:
                completion = client.chat.completions.create(
                    model=self.name,
                    messages=self.message_param,
                    temperature=self.temperature,
                )
                message = completion.choices[0].message
                usage = completion.usage
                # import random
                # sleep(random.randint(1, 5))
                # message = ChatCompletionMessage(content="Debug.", role="assistant")
                # usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
                success = True
                break
            except:
                run_times += 1
        else:
            success = False
            message = ChatCompletionMessage(content="Fail.", role="assistant")
            usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        return message, usage, success


class QueryManager(BaseModel):
    input_dir: str = None
    output_dir: str = None
    name: str = "gpt-3.5-turbo-1106"
    client: Literal["local", "azure"] = "local"
    chunk_size: int = 20
    prompt_debug: bool = False
    failure_limit: int = 10

    _force_fail: bool = False
    _src_files: List[Path]
    _src_dt: Dict[str, List[dict]]
    _result: Dict[str, List[dict]]

    def __init__(self, **data):
        super().__init__(**data)
        assert self.name in SUPPORTED_MODEL_NAMES[self.client], f"Model name {self.name} is not supported by your client."

        assert Path(self.input_dir).exists(), "The specified input directory is invalid!"
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)

        self._src_files = list(Path(self.input_dir).glob("*.jsonl"))
        self._src_dt = {}
        self._result = {}
        assert self._src_files, "No json line files are detected in the input directory!"

        # try to recover from result
        for src in self._src_files:
            possible_result = Path(self.output_dir) / f"result_{src.stem}.jsonl"
            self._src_dt[src.stem] = rjl(src)
            if not possible_result.exists():
                self._result[src.stem] = [{"response": "Fail.", "success": False}] * len(self._src_dt[src.stem])
            else:
                self._result[src.stem] = rjl(possible_result)

    def live_display(self):
        console = Console()
        job_progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[b]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        total_progress = Progress(
            "[b]{task.description}",
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[b]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        total = 0
        prompt_tokens = 0
        completion_tokens = 0
        finished_samples = 0
        failed_samples = 0
        saved_samples = 0
        estimated_price = 0
        price_per_k_tokens = TOKEN_PRICE[MODEL_ALIAS.get(self.name, self.name)]
        task_mapping: bidict[str, TaskID] = bidict({})
        client = load_local_client() if self.client == "local" else load_azure_client()
        if self.prompt_debug:
            enc = tiktoken.encoding_for_model("gpt-4")
        else:
            enc = None

        for k in self._src_dt.keys():
            src_len = len(self._src_dt[k])
            task_mapping[k] = job_progress.add_task(f"[hot_pink b]{k}", total=src_len)
            total += src_len
        overall_task = total_progress.add_task("All Tasks", total=total)

        job_panel = Panel(job_progress, title="[b]Jobs", border_style="deep_sky_blue1", padding=(1, 2))

        def update_status():
            status_style = "[dark_orange b]"
            return (
                f"🚀 {status_style}Prompt Estimation: {'[green1]ON' if self.prompt_debug else '[grey50]OFF'}[/]\n"
                f"🚀 {status_style}Client Type: [green1]{self.client}[/]\n"
                f"🚀 {status_style}Chunk Size: [green1]{self.chunk_size}[/]\n"
                f"🚀 {status_style}OpenAI Model: [green1]{self.name}[/]\n\n"
                f"🚀 {status_style}Prompt Tokens: [green1]{prompt_tokens}[/]\n"
                f"🚀 {status_style}Completion Tokens: [green1]{completion_tokens}[/]\n"
                f"🚀 {status_style}Total Tokens: [green1]{prompt_tokens + completion_tokens}[/]\n"
                f"🚀 {status_style}Estimated Price: {f'[green1]${round(estimated_price / 1000, 2)}' if self.client == 'local' else '[grey50]Disabled'}\n\n"
                f"❗️ {status_style}Finished Samples: [green1]{finished_samples} / {total}[/]\n"
                f"❗️ {status_style}Saved Samples: [green1]{saved_samples} / {total}[/]\n"
                f"❗️ {status_style}Failed Samples: [red1]{failed_samples} / {self.failure_limit}[/]\n"
                f"❗️ {status_style}Force Stop: {'[red1]ON' if self._force_fail else '[grey50]OFF'}[/]"
            )

        status_panel = Panel(
            Text.from_markup(update_status()),
            title="[b]Status",
            border_style="deep_sky_blue1",
            padding=(1, 2),
            width=int(0.35 * console.width),
        )
        detail_table = Table.grid(expand=True)
        detail_table.add_row(status_panel, job_panel)

        vspace = Text("\n")
        display_group = Group(total_progress, vspace, detail_table)
        display_panel = Panel(
            display_group,
            title="[b]Query Dashboard",
            border_style="green3",
            padding=(2, 2),
        )
        temp_result = []

        with Live(display_panel, refresh_per_second=10):
            self._force_fail = False
            while not total_progress.finished:
                mapping = {}
                with ThreadPoolExecutor() as executor:
                    for k, v in self._src_dt.items():
                        task = task_mapping[k]
                        for idv, vv in enumerate(v):
                            # check whether the sample appears in the checkpoint
                            if self._result[k][idv]["success"]:
                                finished_samples += 1
                                job_progress.advance(task)
                                total_progress.advance(overall_task)
                                continue
                            q = Query(
                                name=self.name,
                                debug=self.prompt_debug,
                                force_fail=self._force_fail,
                                **vv,
                            )
                            mapping[executor.submit(q.query, client, enc)] = task, idv
                    for future in as_completed(mapping):
                        tt, idv = mapping[future]
                        job_progress.advance(tt)
                        total_progress.advance(overall_task)

                        message, usage, success = future.result()
                        message: ChatCompletionMessage
                        usage: CompletionUsage
                        if not self.prompt_debug:
                            temp_result.append(
                                {
                                    "source": task_mapping.inverse[tt],
                                    "index": idv,
                                    "response": message.content,
                                }
                            )
                            if len(temp_result) == self.chunk_size:
                                hashed_value = hashlib.sha256(str(f"{temp_result}_{time.time()}").encode()).hexdigest()
                                wjl(
                                    Path(self.output_dir) / f"tmp_{hashed_value}.jsonl",
                                    temp_result,
                                )
                                saved_samples += self.chunk_size
                                temp_result = []

                        prompt_tokens += usage.prompt_tokens
                        completion_tokens += usage.completion_tokens
                        estimated_price += (
                            usage.prompt_tokens * price_per_k_tokens[0] + usage.completion_tokens * price_per_k_tokens[1]
                        )
                        finished_samples += int(success)
                        failed_samples += int(success is False)
                        self._force_fail = failed_samples == self.failure_limit
                        status_panel.renderable = update_status()
            if temp_result:
                hashed_value = hashlib.sha256(str(f"{temp_result}_{time.time()}").encode()).hexdigest()
                wjl(Path(self.output_dir) / f"tmp_{hashed_value}.jsonl", temp_result)
                saved_samples += len(temp_result)
                temp_result = []
                status_panel.renderable = update_status()
        return

    def recover_result(self):
        temp_result_files = list(Path(self.output_dir).glob("tmp_*.jsonl"))
        if temp_result_files:
            for file in temp_result_files:
                chunk = rjl(file)
                for ch in chunk:
                    self._result[ch["source"]][ch["index"]] = {
                        "response": ch["response"],
                        "success": True,
                    }
                file.unlink()
        for k, v in self._result.items():
            wjl(Path(self.output_dir) / f"result_{k}.jsonl", v)
