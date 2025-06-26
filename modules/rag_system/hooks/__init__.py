from .base import BaseHook
from .lower_case import LowerCaseHook
from .remove_stopwords import RemoveStopwordsHook
from .remove_punctuation import RemovePunctuationHook
from .strip_html import StripHTMLHook
from .strip_markdown import StripMarkdownHook
from .remove_empty_lines import RemoveEmptyLinesHook
from .remove_extra_spaces import RemoveExtraSpacesHook
from .custom_replace import CustomReplaceHook

ALL_HOOKS = {
    "lower_case": LowerCaseHook,
    "remove_stopwords": RemoveStopwordsHook,
    "remove_punctuation": RemovePunctuationHook,
    "strip_html": StripHTMLHook,
    "strip_markdown": StripMarkdownHook,
    "remove_empty_lines": RemoveEmptyLinesHook,
    "remove_extra_spaces": RemoveExtraSpacesHook,
    "custom_replace": CustomReplaceHook,
}
