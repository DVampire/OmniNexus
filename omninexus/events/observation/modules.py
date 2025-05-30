from dataclasses import dataclass, field

from browsergym.utils.obs import flatten_axtree_to_str

from omninexus.core.schema import ObservationType
from omninexus.events.observation.observation import Observation


@dataclass
class RelevantResearchRetrievalOutputObservation(Observation):
    """This data class represents the output of a browser."""

    url: str
    screenshot: str = field(repr=False)  # don't show in repr
    error: bool = False
    observation: str = ObservationType.BROWSE
    # do not include in the memory
    open_pages_urls: list = field(default_factory=list)
    active_page_index: int = -1
    dom_object: dict = field(default_factory=dict, repr=False)  # don't show in repr
    axtree_object: dict = field(default_factory=dict, repr=False)  # don't show in repr
    extra_element_properties: dict = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    last_browser_action: str = ''
    last_browser_action_error: str = ''
    focused_element_bid: str = ''

    @property
    def message(self) -> str:
        return 'Visited ' + self.url

    def __str__(self) -> str:
        ret = (
            '**RelevantResearchRetrievalOutputObservation**\n'
            f'URL: {self.url}\n'
            f'Error: {self.error}\n'
            f'Open pages: {self.open_pages_urls}\n'
            f'Active page index: {self.active_page_index}\n'
            f'Last browser action: {self.last_browser_action}\n'
            f'Last browser action error: {self.last_browser_action_error}\n'
            f'Focused element bid: {self.focused_element_bid}\n'
            f'Content: {self.content}\n'
        )
        ret += '--- Agent Observation ---\n'
        ret += self.get_agent_obs_text()
        return ret

    def get_agent_obs_text(self) -> str:
        """Get a concise text that will be shown to the agent."""
        text = f'[Current URL: {self.url}]\n'
        text += f'[Focused element bid: {self.focused_element_bid}]\n\n'
        if self.error:
            text += (
                '================ BEGIN error message ===============\n'
                'The following error occurred when executing the last action:\n'
                f'{self.last_browser_action_error}\n'
                '================ END error message ===============\n'
            )
        else:
            text += '[Action executed successfully.]\n'

        try:
            # We do not filter visible only here because we want to show the full content
            # of the web page to the agent for simplicity.
            cur_axtree_txt = self.get_axtree_str(filter_visible_only=False)
            text += (
                f'============== BEGIN accessibility tree ==============\n'
                f'{cur_axtree_txt}\n'
                f'============== END accessibility tree ==============\n'
            )
        except Exception as e:
            text += f'\n[Error encountered when processing the accessibility tree: {e}]'
        return text

    def get_axtree_str(self, filter_visible_only: bool = False) -> str:
        cur_axtree_txt = flatten_axtree_to_str(
            self.axtree_object,
            extra_properties=self.extra_element_properties,
            with_clickable=True,
            skip_generic=False,
            filter_visible_only=filter_visible_only,
        )
        self._axtree_str = cur_axtree_txt
        return cur_axtree_txt


@dataclass
class IdeaGenerationObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN
    interpreter_details: str = ''

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Idea Generation command `{self.command}` executed with exit code {self.exit_code}.'

    def __str__(self) -> str:
        return f'**IdeaGenerationObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'


@dataclass
class ProjectObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN
    interpreter_details: str = ''

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Project command `{self.command}` executed with exit code {self.exit_code}.'

    def __str__(self) -> str:
        return f'**ProjectObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'


@dataclass
class LatexObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN
    interpreter_details: str = ''

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return (
            f'Latex command `{self.command}` executed with exit code {self.exit_code}.'
        )

    def __str__(self) -> str:
        return f'**LatexObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'


@dataclass
class ReviewObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN
    interpreter_details: str = ''

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Review `{self.command}` executed with exit code {self.exit_code}.'

    def __str__(self) -> str:
        return f'**ReviewObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'
