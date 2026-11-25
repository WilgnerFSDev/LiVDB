from typing import List, Tuple

try:
    import questionary
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:
    questionary = None
    KeyBindings = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except ImportError:
    console = None
    Panel = None
    Text = None


class UI:
    @classmethod
    def clear(cls) -> None:
        if console:
            console.clear()
        else:
            print("\033[H\033[J", end="")

    @classmethod
    def wait_for_user(cls) -> None:
        print("\nPress Enter to continue...")
        input()

    @classmethod
    def header(cls, text: str) -> None:
        if console:
            console.print(f"\n[bold magenta]◆ {text}[/bold magenta]\n")
        else:
            print(f"\n{'=' * 50}\n◆ {text}\n{'=' * 50}\n")

    @classmethod
    def step(cls, text: str) -> None:
        if console:
            console.print(f"[bold cyan]● {text}[/bold cyan]")
        else:
            print(f"● {text}")

    @classmethod
    def success(cls, text: str) -> None:
        if console:
            console.print(f"[bold green]✔ {text}[/bold green]")
        else:
            print(f"✔ {text}")

    @classmethod
    def warning(cls, text: str) -> None:
        if console:
            console.print(f"\n[bold yellow]⚠ {text}[/bold yellow]")
        else:
            print(f"⚠ {text}")

    @classmethod
    def warning_panel(cls, text: str, title: str = "WARNING") -> None:
        if console and Panel:
            panel = Panel(
                f"[bold dark_orange]{text}[/bold dark_orange]",
                title=f"[bold dark_orange]⚠ {title}[/bold dark_orange]",
                border_style="bold dark_orange",
                expand=False,
                padding=(1, 2),
            )
            console.print("")
            console.print(panel)
        else:
            print(f"\n[WARNING: {title}] {text}\n")

    @classmethod
    def error(cls, text: str) -> None:
        if console:
            console.print(f"[bold red]✖ {text}[/bold red]")
        else:
            print(f"✖ {text}")

    @classmethod
    def info(cls, text: str) -> None:
        if console:
            console.print(f"[dim]{text}[/dim]")
        else:
            print(f"  {text}")

    @classmethod
    def ask_string(cls, prompt_text: str, default: str = "") -> str:
        result = questionary.text(prompt_text, default=default).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    @classmethod
    def ask_boolean(cls, prompt_text: str, default: bool = True) -> bool:
        result = questionary.confirm(prompt_text, default=default).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    @classmethod
    def ask_choice(cls, prompt_text: str, choices: list) -> str:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from questionary.constants import DEFAULT_QUESTION_PREFIX, DEFAULT_STYLE
        from questionary.prompts.common import InquirerControl, create_inquirer_layout

        title_map = {}
        q_choices = []
        for choice in choices:
            if isinstance(choice, (tuple, list)):
                key = str(choice[0])
                title = choice[1] if len(choice) > 1 else str(choice[0])
                desc = choice[2] if len(choice) > 2 else ""
            elif isinstance(choice, questionary.Choice):
                key = str(choice.value)
                title = (
                    choice.title
                    if isinstance(choice.title, str)
                    else str(choice.value)
                )
                desc = choice.description or ""
            else:
                key = str(choice)
                title = str(choice)
                desc = ""

            title_map[key] = title

            if desc:
                title_tokens = [
                    ("", f"{key} - {title}\n"),
                    ("class:choice-desc", f"    {desc}"),
                ]
                q_choices.append(
                    questionary.Choice(
                        title=title_tokens, value=key, description=None
                    )
                )
            else:
                q_choices.append(
                    questionary.Choice(
                        title=f"{key} - {title}", value=key, description=None
                    )
                )

        ic = InquirerControl(q_choices)

        def get_prompt_tokens():
            tokens = [
                ("class:qmark", DEFAULT_QUESTION_PREFIX),
                ("class:question", f" {prompt_text} "),
            ]
            if ic.is_answered:
                pointed = ic.get_pointed_at()
                if pointed is not None:
                    selected_val = str(pointed.value)
                    selected_title = title_map.get(selected_val, selected_val)
                    tokens.append(("class:answer", selected_title))
            return tokens

        layout = create_inquirer_layout(ic, get_prompt_tokens)

        kb = KeyBindings()
        shortcuts = {
            str(choice[0])
            if isinstance(choice, (tuple, list))
            else (
                str(choice.value)
                if isinstance(choice, questionary.Choice)
                else str(choice)
            ): str(choice[0])
            if isinstance(choice, (tuple, list))
            else (
                str(choice.value)
                if isinstance(choice, questionary.Choice)
                else str(choice)
            )
            for choice in choices
        }

        @kb.add("0")
        @kb.add("1")
        @kb.add("2")
        @kb.add("3")
        @kb.add("4")
        @kb.add("5")
        @kb.add("6")
        @kb.add("7")
        @kb.add("8")
        @kb.add("9")
        def _(event):
            key = event.key_sequence[0].data
            if key in shortcuts:
                target_val = shortcuts[key]
                for idx, c in enumerate(ic.choices):
                    if (
                        getattr(c, "value", None) == target_val
                        or str(getattr(c, "value", None)) == target_val
                    ):
                        ic.pointed_at = idx
                        break
                ic.is_answered = True
                event.app.exit(result=target_val)

        @kb.add("up")
        def _(event):
            ic.select_previous()

        @kb.add("down")
        def _(event):
            ic.select_next()

        @kb.add("enter")
        def _(event):
            ic.is_answered = True
            event.app.exit(result=ic.get_pointed_at().value)

        @kb.add("c-c")
        @kb.add("c-q")
        def _(event):
            event.app.exit(result=None)

        from prompt_toolkit.styles import Style, merge_styles

        custom_style = merge_styles(
            [
                DEFAULT_STYLE,
                Style(
                    [
                        ("choice-desc", "fg:#777777 italic"),
                    ]
                ),
            ]
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=custom_style,
            full_screen=False,
        )

        result = app.run()

        if result is None:
            raise KeyboardInterrupt
        return result

    @classmethod
    def print_banner(cls) -> None:
        banner_text = Text(justify="center")

        banner_text.append("LiVDB", style="bold magenta")
        banner_text.append(
            " - Project Vulnerabilities Dataset Builder\n\n", style="bold white"
        )

        banner_text.append(
            "Automated pipeline for dataset generation\n", style="dim white"
        )
        banner_text.append(
            "from Project source code vulnerabilities.\n\n", style="dim white"
        )

        banner_text.append("Author: Wilgner Fabio da Silva - ", style="cyan")
        banner_text.append("wilgner.fabio@aluno.ifsp.edu.br\n", style="underline cyan")
        banner_text.append("Advisor: Kleber Manrique Trevisani - ", style="cyan")
        banner_text.append("kleber@ifsp.edu.br\n", style="underline cyan")

        banner_text.append(
            "IFSP - Instituto Federal de Educação, Ciência e Tecnologia de São Paulo\n",
            style="green",
        )
        banner_text.append("B.Sc. in Computer Science", style="green")

        panel = Panel(
            banner_text, border_style="bold cyan", expand=False, padding=(1, 4)
        )
        console.print(panel)
        console.print(
            "\n[dim]Tip: Press CTRL+C at any time to cancel an action and return to the menu.[/dim]\n"
        )
