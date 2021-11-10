#!/usr/bin/env python3
"""Tokenizes, verbalizes, and phonemizes text and SSML"""
import itertools
import logging
import re
import typing
import xml.etree.ElementTree as etree
from decimal import Decimal
from pathlib import Path

import babel
import babel.numbers
import dateparser
import networkx as nx
from gruut_ipa import IPA
from num2words import num2words

from gruut.const import (
    DATA_PROP,
    PHONEMES_TYPE,
    REGEX_PATTERN,
    BreakNode,
    BreakType,
    BreakWordNode,
    EndElement,
    GraphType,
    IgnoreNode,
    InlineLexicon,
    InterpretAs,
    InterpretAsFormat,
    Lexeme,
    MarkNode,
    Node,
    ParagraphNode,
    PunctuationWordNode,
    Sentence,
    SentenceNode,
    SpeakNode,
    SSMLParsingState,
    TextProcessorSettings,
    Word,
    WordNode,
    WordRole,
)
from gruut.lang import get_settings
from gruut.utils import (
    attrib_no_namespace,
    leaves,
    maybe_split_ipa,
    pipeline_split,
    pipeline_transform,
    resolve_lang,
    tag_no_namespace,
    text_and_elements,
)

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut.text_processor")

DEFAULT_LEXICON_ID = ""


# -----------------------------------------------------------------------------


class TextProcessor:
    """Tokenizes, verbalizes, and phonemizes text and SSML"""

    def __init__(
        self,
        default_lang: str = "en_US",
        model_prefix: str = "",
        lang_dirs: typing.Optional[typing.Dict[str, typing.Union[str, Path]]] = None,
        search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
        settings: typing.Optional[
            typing.MutableMapping[str, TextProcessorSettings]
        ] = None,
        **kwargs,
    ):
        self.default_lang = default_lang
        self.default_settings_kwargs = kwargs

        self.model_prefix = model_prefix
        self.search_dirs = search_dirs

        if lang_dirs is None:
            lang_dirs = {}

        # Convert to Paths
        self.lang_dirs = {
            dir_lang: Path(dir_path) for dir_lang, dir_path in lang_dirs.items()
        }

        if settings is None:
            settings = {}

        self.settings = settings

    def sentences(
        self,
        graph: GraphType,
        root: Node,
        major_breaks: bool = True,
        minor_breaks: bool = True,
        punctuations: bool = True,
        explicit_lang: bool = True,
        phonemes: bool = True,
        break_phonemes: bool = True,
        pos: bool = True,
    ) -> typing.Iterable[Sentence]:
        """Processes text and returns each sentence"""

        def get_lang(lang: str) -> str:
            if explicit_lang or (lang != self.default_lang):
                return lang

            # Implicit default language
            return ""

        sentence: typing.Optional[Sentence] = None

        par_idx: int = -1
        sent_idx: int = 0

        sent_pause_before_ms: int = 0
        sent_marks_before: typing.List[str] = []

        word_pause_before_ms: int = 0
        word_marks_before: typing.List[str] = []

        sentences: typing.List[Sentence] = []

        for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
            node = graph.nodes[dfs_node][DATA_PROP]
            if isinstance(node, ParagraphNode):
                par_idx += 1
                sent_idx = 0
            elif isinstance(node, SentenceNode):
                # New sentence
                sentences.append(
                    Sentence(
                        idx=sent_idx,
                        par_idx=par_idx,
                        text="",
                        text_with_ws="",
                        text_spoken="",
                        voice=node.voice,
                        lang=get_lang(node.lang),
                        pause_before_ms=sent_pause_before_ms,
                        marks_before=(sent_marks_before if sent_marks_before else None),
                    )
                )

                sent_pause_before_ms = 0
                sent_marks_before = []
                sent_idx += 1
            elif graph.out_degree(dfs_node) == 0:
                if isinstance(node, WordNode):
                    assert sentences, "No sentence"
                    sentence = sentences[-1]

                    word_node = typing.cast(WordNode, node)
                    sentence.words.append(
                        Word(
                            idx=len(sentence.words),
                            sent_idx=sentence.idx,
                            par_idx=sentence.par_idx,
                            text=word_node.text,
                            text_with_ws=word_node.text_with_ws,
                            phonemes=word_node.phonemes if phonemes else None,
                            pos=word_node.pos if pos else None,
                            lang=get_lang(node.lang),
                            voice=node.voice,
                            pause_before_ms=word_pause_before_ms,
                            marks_before=(
                                word_marks_before if word_marks_before else None
                            ),
                        )
                    )

                    word_pause_before_ms = 0
                    word_marks_before = []
                elif isinstance(node, BreakWordNode):
                    assert sentences, "No sentence"
                    sentence = sentences[-1]

                    break_word_node = typing.cast(BreakWordNode, node)
                    is_minor_break = break_word_node.break_type == BreakType.MINOR
                    is_major_break = break_word_node.break_type == BreakType.MAJOR

                    if (minor_breaks and is_minor_break) or (
                        major_breaks and is_major_break
                    ):
                        sentence.words.append(
                            Word(
                                idx=len(sentence.words),
                                sent_idx=sentence.idx,
                                par_idx=sentence.par_idx,
                                text=break_word_node.text,
                                text_with_ws=break_word_node.text_with_ws,
                                phonemes=self._phonemes_for_break(
                                    break_word_node.break_type,
                                    lang=break_word_node.lang,
                                )
                                if phonemes and break_phonemes
                                else None,
                                is_minor_break=is_minor_break,
                                is_major_break=is_major_break,
                                lang=get_lang(node.lang),
                                voice=node.voice,
                                pause_before_ms=word_pause_before_ms,
                                marks_before=(
                                    word_marks_before if word_marks_before else None
                                ),
                            )
                        )

                        word_pause_before_ms = 0
                        word_marks_before = []
                elif punctuations and isinstance(node, PunctuationWordNode):
                    assert sentences, "No sentence"
                    sentence = sentences[-1]

                    punct_word_node = typing.cast(PunctuationWordNode, node)
                    sentence.words.append(
                        Word(
                            idx=len(sentence.words),
                            sent_idx=sentence.idx,
                            par_idx=sentence.par_idx,
                            text=punct_word_node.text,
                            text_with_ws=punct_word_node.text_with_ws,
                            is_punctuation=True,
                            lang=get_lang(punct_word_node.lang),
                            pause_before_ms=word_pause_before_ms,
                            marks_before=(
                                word_marks_before if word_marks_before else None
                            ),
                        )
                    )

                    word_pause_before_ms = 0
                    word_marks_before = []
                elif isinstance(node, BreakNode):
                    # Pause for some time
                    break_node = typing.cast(BreakNode, node)
                    break_parent = self._find_parent(
                        graph, node, (SentenceNode, ParagraphNode, SpeakNode)
                    )

                    if break_parent is not None:
                        break_ms = break_node.get_milliseconds()
                        break_parent_edges = list(graph.out_edges(break_parent.node))
                        break_edge_idx = break_parent_edges.index(
                            (break_parent.node, break_node.node)
                        )
                        is_last_edge = break_edge_idx == (len(break_parent_edges) - 1)

                        if isinstance(break_parent, SentenceNode):
                            assert sentences
                            sentence = sentences[-1]
                            if is_last_edge:
                                # End of sentence, add pause after
                                sentence.pause_after_ms += break_ms
                            elif sentence.words:
                                # Between words, add pause after previous word
                                sentence.words[-1].pause_after_ms += break_ms
                            else:
                                # Before first word, set pause for first word
                                word_pause_before_ms += break_ms
                        elif isinstance(break_parent, ParagraphNode):
                            if sentences and (sentences[-1].par_idx == par_idx):
                                # Between sentences in the same paragraph, add pause after previous sentence
                                sentences[-1].pause_after_ms += break_ms
                            else:
                                # Add pause to beginning of next sentence
                                sent_pause_before_ms += break_ms
                        elif isinstance(break_parent, SpeakNode):
                            if sentences:
                                # After paragraphs or sentences
                                sentences[-1].pause_after_ms += break_ms
                            else:
                                # Before any paragraphs or sentences
                                sent_pause_before_ms += break_ms
                elif isinstance(node, MarkNode):
                    # User-defined mark
                    mark_node = typing.cast(MarkNode, node)
                    mark_name = mark_node.name
                    mark_parent = self._find_parent(
                        graph, node, (SentenceNode, ParagraphNode, SpeakNode)
                    )

                    if mark_parent is not None:
                        mark_parent_edges = list(graph.out_edges(mark_parent.node))
                        mark_edge_idx = mark_parent_edges.index(
                            (mark_parent.node, mark_node.node)
                        )
                        is_last_edge = mark_edge_idx == (len(mark_parent_edges) - 1)

                        if isinstance(mark_parent, SentenceNode):
                            assert sentences
                            sentence = sentences[-1]
                            if is_last_edge:
                                # End of sentence, add mark after
                                if sentence.marks_after is None:
                                    sentence.marks_after = []

                                sentence.marks_after.append(mark_name)
                            elif sentence.words:
                                # Between words, add pause after previous word
                                last_word = sentence.words[-1]
                                if last_word.marks_after is None:
                                    last_word.marks_after = []

                                last_word.marks_after.append(mark_name)
                            else:
                                # Before first word, set pause for first word
                                word_marks_before.append(mark_name)
                        elif isinstance(mark_parent, ParagraphNode):
                            if sentences and (sentences[-1].par_idx == par_idx):
                                # Between sentences in the same paragraph, add pause after previous sentence
                                last_sentence = sentences[-1]
                                if last_sentence.marks_after is None:
                                    last_sentence.marks_after = []

                                last_sentence.marks_after.append(mark_name)
                            else:
                                # Add pause to beginning of next sentence
                                sent_marks_before.append(mark_name)
                        elif isinstance(mark_parent, SpeakNode):
                            if sentences:
                                # After paragraphs or sentences
                                last_sentence = sentences[-1]
                                if last_sentence.marks_after is None:
                                    last_sentence.marks_after = []

                                last_sentence.marks_after.append(mark_name)
                            else:
                                # Before any paragraphs or sentences
                                sent_marks_before.append(mark_name)

        # Post-process sentences to fix up text, voice, etc.
        for sentence in sentences:
            settings = self.get_settings(sentence.lang)
            if settings.keep_whitespace:
                # Whitespace is preseved
                sentence.text_with_ws = "".join(w.text_with_ws for w in sentence.words)
            else:
                # Make a best guess.
                # The join string is used before spoken words (except the first word).
                # This should have the effect of keeping punctuation next to words.
                word_texts: typing.List[str] = []
                for word in sentence.words:
                    if word.is_spoken:
                        if word_texts:
                            word_texts.append(f"{settings.join_str}{word.text}")
                        else:
                            word_texts.append(word.text)
                    else:
                        word_texts.append(word.text)

                sentence.text_with_ws = "".join(word_texts)

            sentence.text = settings.normalize_whitespace(sentence.text_with_ws)
            sentence.text_spoken = settings.join_str.join(
                w.text for w in sentence.words if w.is_spoken
            )

            # Normalize voice
            sent_voice = sentence.voice

            # Get voice used across all words
            for word in sentence.words:
                if word.voice:
                    if sent_voice and (sent_voice != word.voice):
                        # Multiple voices
                        sent_voice = ""
                        break

                    sent_voice = word.voice

            if sent_voice:
                sentence.voice = sent_voice

                # Set voice on all words
                for word in sentence.words:
                    word.voice = sent_voice

        return sentences

    def words(self, graph: GraphType, root: Node, **kwargs) -> typing.Iterable[Word]:
        """Processes text and returns each word"""
        for sent in self.sentences(graph, root, **kwargs):
            for word in sent:
                yield word

    def get_settings(self, lang: typing.Optional[str] = None) -> TextProcessorSettings:
        """Gets or creates settings for a language"""
        lang = lang or self.default_lang
        lang_settings = self.settings.get(lang)

        if lang_settings is not None:
            return lang_settings

        # Try again with resolved language
        resolved_lang = resolve_lang(lang)
        lang_settings = self.settings.get(resolved_lang)
        if lang_settings is not None:
            # Patch for the future
            self.settings[lang] = self.settings[resolved_lang]
            return lang_settings

        _LOGGER.debug(
            "No custom settings for language %s (%s). Creating default settings.",
            lang,
            resolved_lang,
        )

        # Create default settings for language
        lang_dir = self.lang_dirs.get(lang)
        lang_settings = get_settings(
            lang,
            lang_dir=lang_dir,
            model_prefix=self.model_prefix,
            search_dirs=self.search_dirs,
            **self.default_settings_kwargs,
        )
        self.settings[lang] = lang_settings
        self.settings[resolved_lang] = lang_settings

        return lang_settings

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Processes text or SSML"""
        return self.process(*args, **kwargs)

    def process(
        self,
        text: str,
        lang: typing.Optional[str] = None,
        ssml: bool = False,
        pos: bool = True,
        phonemize: bool = True,
        post_process: bool = True,
        add_speak_tag: bool = True,
        detect_numbers: bool = True,
        detect_currency: bool = True,
        detect_dates: bool = True,
        detect_times: bool = True,
        verbalize_numbers: bool = True,
        verbalize_currency: bool = True,
        verbalize_dates: bool = True,
        verbalize_times: bool = True,
        max_passes: int = 5,
    ) -> typing.Tuple[GraphType, Node]:
        """
        Processes text or SSML

        Args:
            text: input text or SSML (ssml=True)
            lang: default language of input text
            ssml: True if input text is SSML
            pos: False if part of speech tagging should be disabled
            phonemize: False if phonemization should be disabled
            post_process: False if sentence/graph post-processing should be disabled
            add_speak_tag: True if <speak> should be automatically added to input text when ssml=True
            detect_numbers: True if numbers should be annotated in text (interpret_as="number")
            detect_currency: True if currency amounts should be annotated in text (interpret_as="currency")
            detect_dates: True if dates should be annotated in text (interpret_as="date")
            detect_times: True if clock times should be annotated in text (interpret_as="time")
            verbalize_numbers: True if annotated numbers should be expanded into words
            verbalize_currency: True if annotated currency amounts should be expanded into words
            verbalize_dates: True if annotated dates should be expanded into words
            verbalize_times: True if annotated clock times should be expanded into words

        Returns:
            graph, root: text graph and root node

        """
        if ssml:
            try:
                root_element = etree.fromstring(text)
            except Exception as e:
                if add_speak_tag:
                    # Try wrapping text in <speak> and parsing again
                    root_element = etree.fromstring(f"<speak>{text}</speak>")
                else:
                    # Log and re-raise exception
                    _LOGGER.exception("TextProcessor.process")
                    raise e

            def iter_elements():
                yield from text_and_elements(root_element)

        else:
            # Not XML
            def iter_elements():
                yield text

        graph = typing.cast(GraphType, nx.DiGraph())

        # Parse XML
        last_paragraph: typing.Optional[ParagraphNode] = None
        last_sentence: typing.Optional[SentenceNode] = None
        last_speak: typing.Optional[SpeakNode] = None
        root: typing.Optional[SpeakNode] = None
        parsing_state = SSMLParsingState.DEFAULT

        # [voice]
        voice_stack: typing.List[str] = []

        # [(interpret_as, format)]
        say_as_stack: typing.List[typing.Tuple[str, str]] = []

        # [(tag, lang)]
        lang_stack: typing.List[typing.Tuple[str, str]] = []
        current_lang: str = lang or self.default_lang

        # [lexicon.id]
        lookup_stack: typing.List[str] = []
        lexicon_id: typing.Optional[str] = None
        lexeme: typing.Optional[Lexeme] = None

        # id -> lexicon
        inline_lexicons: typing.Dict[str, InlineLexicon] = {}

        # True if current word is the last one
        is_last_word: bool = False

        # Current word's role
        word_role: typing.Optional[str] = None

        # Alias from <sub>
        last_alias: typing.Optional[str] = None

        # Used to skip <metadata>
        skip_elements: bool = False

        # Phonemes to use for next word(s)
        word_phonemes: typing.Optional[typing.List[typing.List[str]]] = None

        # Create __init__ args for new Node
        def scope_kwargs(target_class):
            scope = {}
            if voice_stack:
                scope["voice"] = voice_stack[-1]

            scope["lang"] = current_lang

            if target_class is WordNode:
                if say_as_stack:
                    scope["interpret_as"], scope["format"] = say_as_stack[-1]

                if word_role is not None:
                    scope["role"] = word_role

                if lookup_stack:
                    # Lexicon ids in order of look up
                    scope["lexicon_ids"] = list(reversed(lookup_stack))

            return scope

        def in_inline_lexicon(
            word_text: str, word_role: typing.Optional[str] = None
        ) -> bool:
            if inline_lexicons:
                for inline_lexicon_id in itertools.chain(
                    lookup_stack, [DEFAULT_LEXICON_ID]
                ):
                    maybe_lexicon = inline_lexicons.get(inline_lexicon_id)
                    if maybe_lexicon is None:
                        continue

                    maybe_role_phonemes = maybe_lexicon.words.get(word_text)
                    if maybe_role_phonemes is None:
                        continue

                    if (word_role is not None) and (word_role in maybe_role_phonemes):
                        # Role-specific pronunciation
                        return True

                    if WordRole.DEFAULT in maybe_role_phonemes:
                        # Default pronunciation
                        return True

            # No inline pronunciation
            return False

        # Process sub-elements and text chunks
        for elem_or_text in iter_elements():
            if isinstance(elem_or_text, str):
                if skip_elements:
                    # Inside <metadata>
                    continue

                # Text chunk
                text = typing.cast(str, elem_or_text)

                # <grapheme> inside <lexicon>
                if parsing_state == SSMLParsingState.IN_LEXICON_GRAPHEME:
                    assert lexeme is not None
                    lexeme.grapheme = text.strip()
                    continue

                # <phoneme> inside <lexicon>
                if parsing_state == SSMLParsingState.IN_LEXICON_PHONEME:
                    assert lexeme is not None
                    text = text.strip()

                    # Phonemes will be split on whitespace if at least one
                    # space is present, otherwise assume phonemes =
                    # graphemes.
                    lexeme.phonemes = maybe_split_ipa(text)
                    continue

                if last_alias is not None:
                    # Iniside a <sub>
                    text = last_alias

                if last_speak is None:
                    # Implicit <speak>
                    last_speak = SpeakNode(node=len(graph), implicit=True)
                    graph.add_node(last_speak.node, data=last_speak)
                    if root is None:
                        root = last_speak

                assert last_speak is not None

                if last_paragraph is None:
                    # Implicit <p>
                    p_node = ParagraphNode(
                        node=len(graph), implicit=True, **scope_kwargs(ParagraphNode)
                    )
                    graph.add_node(p_node.node, data=p_node)

                    graph.add_edge(last_speak.node, p_node.node)
                    last_paragraph = p_node

                assert last_paragraph is not None

                if last_sentence is None:
                    # Implicit <s>
                    s_node = SentenceNode(
                        node=len(graph), implicit=True, **scope_kwargs(SentenceNode)
                    )
                    graph.add_node(s_node.node, data=s_node)

                    graph.add_edge(last_paragraph.node, s_node.node)
                    last_sentence = s_node

                assert last_sentence is not None

                if parsing_state == SSMLParsingState.IN_WORD:
                    # No splitting
                    word_text = text
                    settings = self.get_settings(current_lang)
                    if (
                        settings.keep_whitespace
                        and (not is_last_word)
                        and (not word_text.endswith(settings.join_str))
                    ):
                        word_text += settings.join_str

                    word_kwargs = scope_kwargs(WordNode)
                    if word_phonemes:
                        word_kwargs["phonemes"] = word_phonemes.pop()

                    word_text_norm = settings.normalize_whitespace(word_text)

                    word_node = WordNode(
                        node=len(graph),
                        text=word_text_norm,
                        text_with_ws=word_text,
                        in_lexicon=(
                            in_inline_lexicon(word_text_norm, word_role)
                            or self._is_word_in_lexicon(word_text_norm, settings)
                        ),
                        **word_kwargs,
                    )
                    graph.add_node(word_node.node, data=word_node)
                    graph.add_edge(last_sentence.node, word_node.node)
                else:
                    # Split by whitespace
                    self._pipeline_tokenize(
                        graph,
                        last_sentence,
                        text,
                        word_phonemes=word_phonemes,
                        scope_kwargs=scope_kwargs(WordNode),
                        in_inline_lexicon=in_inline_lexicon,
                    )

            elif isinstance(elem_or_text, EndElement):
                # End of an element (e.g., </s>)
                end_elem = typing.cast(EndElement, elem_or_text)
                end_tag = tag_no_namespace(end_elem.element.tag)

                if end_tag == "voice":
                    if voice_stack:
                        voice_stack.pop()
                elif end_tag == "say-as":
                    if say_as_stack:
                        say_as_stack.pop()
                elif end_tag == "lookup":
                    if lookup_stack:
                        lookup_stack.pop()
                elif end_tag == "lexicon":
                    # Done parsing <lexicon>
                    parsing_state = SSMLParsingState.DEFAULT
                    lexicon_id = None
                elif (end_tag == "grapheme") and (
                    parsing_state == SSMLParsingState.IN_LEXICON_GRAPHEME
                ):
                    # Done with lexicon grapheme
                    parsing_state = SSMLParsingState.IN_LEXICON
                elif (end_tag == "phoneme") and (
                    parsing_state == SSMLParsingState.IN_LEXICON_PHONEME
                ):
                    # Done with lexicon phoneme
                    parsing_state = SSMLParsingState.IN_LEXICON
                elif (end_tag == "lexeme") and (
                    parsing_state == SSMLParsingState.IN_LEXICON
                ):
                    # Done with lexicon entry
                    assert lexeme is not None, "No lexeme"
                    assert (
                        lexeme.phonemes is not None
                    ), f"No phoneme for lexeme: {lexeme}"

                    assert lexicon_id is not None, "No lexicon id"
                    lexicon = inline_lexicons.get(lexicon_id)
                    assert lexicon is not None, f"No lexicon for id {lexicon_id}"

                    # Get or create role -> phonemes map
                    role_phonemes: typing.Dict[str, PHONEMES_TYPE] = lexicon.words.get(
                        lexeme.grapheme, {}
                    )

                    if lexeme.roles:
                        # Add phonemes for each role
                        for role in lexeme.roles:
                            role_phonemes[role] = lexeme.phonemes
                    else:
                        # Default (empty) role only
                        role_phonemes[WordRole.DEFAULT] = lexeme.phonemes

                    lexicon.words[lexeme.grapheme] = role_phonemes

                    # Reset state
                    lexeme = None
                else:
                    if lang_stack and (lang_stack[-1][0] == end_tag):
                        lang_stack.pop()

                    if lang_stack:
                        current_lang = lang_stack[-1][1]  # tag, lang
                    else:
                        current_lang = self.default_lang

                    if end_tag in {"w", "token"}:
                        # End of word
                        parsing_state = SSMLParsingState.DEFAULT
                        is_last_word = False
                        word_role = None
                    elif end_tag == "s":
                        # End of sentence
                        last_sentence = None
                    elif end_tag == "p":
                        # End of paragraph
                        last_paragraph = None
                    elif end_tag == "speak":
                        # End of speak
                        last_speak = root
                    elif end_tag == "sub":
                        # End of sub
                        last_alias = None
                    elif end_tag in {"metadata", "meta"}:
                        # End of metadata
                        skip_elements = False
                    elif end_tag == "phoneme":
                        # End of phoneme
                        word_phonemes = None
            else:
                if skip_elements:
                    # Inside <metadata>
                    continue

                # Start of an element (e.g., <p>)
                elem, elem_metadata = elem_or_text
                elem = typing.cast(etree.Element, elem)

                # Optional metadata for the element
                elem_metadata = typing.cast(
                    typing.Optional[typing.Dict[str, typing.Any]], elem_metadata
                )

                elem_tag = tag_no_namespace(elem.tag)

                if elem_tag == "speak":
                    # Explicit <speak>
                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    speak_node = SpeakNode(
                        node=len(graph), element=elem, **scope_kwargs(SpeakNode)
                    )
                    if root is None:
                        root = speak_node

                    graph.add_node(speak_node.node, data=root)
                    last_speak = root
                elif elem_tag == "voice":
                    # Set voice scope
                    voice_name = attrib_no_namespace(elem, "name")
                    voice_stack.append(voice_name)
                elif elem_tag == "p":
                    # Explicit paragraph
                    if last_speak is None:
                        # Implicit <speak>
                        last_speak = SpeakNode(node=len(graph), implicit=True)
                        graph.add_node(last_speak.node, data=last_speak)
                        if root is None:
                            root = last_speak

                    assert last_speak is not None

                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    p_node = ParagraphNode(
                        node=len(graph), element=elem, **scope_kwargs(ParagraphNode)
                    )
                    graph.add_node(p_node.node, data=p_node)
                    graph.add_edge(last_speak.node, p_node.node)
                    last_paragraph = p_node

                    # Force a new sentence to begin
                    last_sentence = None
                elif elem_tag == "s":
                    # Explicit sentence
                    if last_speak is None:
                        # Implicit <speak>
                        last_speak = SpeakNode(node=len(graph), implicit=True)
                        graph.add_node(last_speak.node, data=last_speak)
                        if root is None:
                            root = last_speak

                    assert last_speak is not None

                    if last_paragraph is None:
                        # Implicit paragraph
                        p_node = ParagraphNode(
                            node=len(graph), **scope_kwargs(ParagraphNode)
                        )
                        graph.add_node(p_node.node, data=p_node)

                        graph.add_edge(last_speak.node, p_node.node)
                        last_paragraph = p_node

                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    s_node = SentenceNode(
                        node=len(graph), element=elem, **scope_kwargs(SentenceNode)
                    )
                    graph.add_node(s_node.node, data=s_node)
                    graph.add_edge(last_paragraph.node, s_node.node)
                    last_sentence = s_node
                elif elem_tag in {"w", "token"}:
                    # Explicit word
                    parsing_state = SSMLParsingState.IN_WORD
                    is_last_word = (
                        elem_metadata.get("is_last", False) if elem_metadata else False
                    )
                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    word_role = attrib_no_namespace(elem, "role")
                elif elem_tag == "break":
                    # Break
                    last_target = last_sentence or last_paragraph or last_speak
                    assert last_target is not None
                    break_node = BreakNode(
                        node=len(graph),
                        element=elem,
                        time=attrib_no_namespace(elem, "time", ""),
                    )
                    graph.add_node(break_node.node, data=break_node)
                    graph.add_edge(last_target.node, break_node.node)
                elif elem_tag == "mark":
                    # Mark
                    last_target = last_sentence or last_paragraph or last_speak
                    assert last_target is not None
                    mark_node = MarkNode(
                        node=len(graph),
                        element=elem,
                        name=attrib_no_namespace(elem, "name", ""),
                    )
                    graph.add_node(mark_node.node, data=mark_node)
                    graph.add_edge(last_target.node, mark_node.node)
                elif elem_tag == "say-as":
                    say_as_stack.append(
                        (
                            attrib_no_namespace(elem, "interpret-as", ""),
                            attrib_no_namespace(elem, "format", ""),
                        )
                    )
                elif elem_tag == "sub":
                    # Sub
                    last_alias = attrib_no_namespace(elem, "alias", "")
                elif elem_tag in {"metadata", "meta"}:
                    # Metadata
                    skip_elements = True
                elif (elem_tag == "phoneme") and (
                    parsing_state != SSMLParsingState.IN_LEXICON
                ):
                    # Phonemes
                    word_phonemes_strs = attrib_no_namespace(elem, "ph", "").split()

                    if word_phonemes_strs:
                        # Phonemes will be split on whitespace if at least one
                        # space is present, otherwise assume phonemes =
                        # graphemes.
                        word_phonemes = [
                            maybe_split_ipa(phoneme_str)
                            for phoneme_str in word_phonemes_strs
                        ]
                    else:
                        word_phonemes = None
                elif elem_tag == "lang":
                    # Set language
                    maybe_lang = attrib_no_namespace(elem, "lang", "")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag == "lookup":
                    lookup_id = attrib_no_namespace(elem, "ref")
                    assert lookup_id is not None, f"Lookup id required ({elem})"
                    lookup_stack.append(lookup_id)
                elif elem_tag == "lexicon":
                    # Inline pronunciaton lexicon
                    # NOTE: Empty lexicon id means the "default" inline lexicon (lookup not required)
                    lexicon_id = attrib_no_namespace(elem, "id", DEFAULT_LEXICON_ID)
                    assert lexicon_id is not None

                    parsing_state = SSMLParsingState.IN_LEXICON
                    lexicon_alphabet = (
                        attrib_no_namespace(elem, "alphabet", "").strip().lower()
                    )
                    inline_lexicons[lexicon_id] = InlineLexicon(
                        lexicon_id=lexicon_id, alphabet=lexicon_alphabet
                    )
                elif (elem_tag == "grapheme") and (
                    parsing_state == SSMLParsingState.IN_LEXICON
                ):
                    # Inline pronunciaton lexicon (grapheme)
                    parsing_state = SSMLParsingState.IN_LEXICON_GRAPHEME
                    if lexeme is None:
                        lexeme = Lexeme()

                    role_str = attrib_no_namespace(elem, "role")
                    if role_str:
                        lexeme.roles = set(role_str.strip().split())
                elif (elem_tag == "phoneme") and (
                    parsing_state == SSMLParsingState.IN_LEXICON
                ):
                    # Inline pronunciaton lexicon (phoneme)
                    parsing_state = SSMLParsingState.IN_LEXICON_PHONEME
                    if lexeme is None:
                        lexeme = Lexeme()

        assert root is not None

        # Do multiple passes over the graph
        num_passes_left = max_passes
        while num_passes_left > 0:
            was_changed = False

            # Do replacements before minor/major breaks
            if pipeline_split(self._split_replacements, graph, root):
                was_changed = True

            # Split punctuations (quotes, etc.) before breaks
            if pipeline_split(self._split_punctuations, graph, root):
                was_changed = True

            # Split on minor breaks (commas, etc.)
            if pipeline_split(self._split_minor_breaks, graph, root):
                was_changed = True

            # Expand abbrevations before major breaks
            if pipeline_split(self._split_abbreviations, graph, root):
                was_changed = True

            # Break apart initialisms (e.g., TTS or T.T.S.) before major breaks
            if pipeline_split(self._split_initialism, graph, root):
                was_changed = True

            # Split on major breaks (periods, etc.)
            if pipeline_split(self._split_major_breaks, graph, root):
                was_changed = True

            # Break apart sentences using BreakWordNodes
            if self._break_sentences(graph, root):
                was_changed = True

            # spell-out (e.g., abc -> a b c) before number expansion
            if pipeline_split(self._split_spell_out, graph, root):
                was_changed = True

            # Transform text into known classes.
            #
            # The order here is very important, since words with "interpret_as"
            # set will be skipped by later transformations.
            #
            # Dates are detected first so words like "1.1.2000" are not parsed
            # as numbers by Babel (the de_DE locale will parse this as 112000).
            #
            if detect_dates:
                if pipeline_transform(self._transform_date, graph, root):
                    was_changed = True

            if detect_currency:
                if pipeline_transform(self._transform_currency, graph, root):
                    was_changed = True

            if detect_numbers:
                if pipeline_transform(self._transform_number, graph, root):
                    was_changed = True

            if detect_times:
                if pipeline_transform(self._transform_time, graph, root):
                    was_changed = True

            # Verbalize known classes
            if verbalize_dates:
                if pipeline_transform(self._verbalize_date, graph, root):
                    was_changed = True

            if verbalize_times:
                if pipeline_transform(self._verbalize_time, graph, root):
                    was_changed = True

            if verbalize_numbers:
                if pipeline_transform(self._verbalize_number, graph, root):
                    was_changed = True

            if verbalize_currency:
                if pipeline_transform(self._verbalize_currency, graph, root):
                    was_changed = True

            # Break apart words
            if pipeline_split(self._break_words, graph, root):
                was_changed = True

            # Ignore non-words
            if pipeline_split(self._split_ignore_non_words, graph, root):
                was_changed = True

            if not was_changed:
                # No changes, so we can stop
                break

            num_passes_left -= 1

        # Gather words from leaves of the tree, group by sentence
        def process_sentence(words: typing.List[WordNode]):
            if pos:
                pos_settings = self.get_settings(node.lang)
                if pos_settings.get_parts_of_speech is not None:
                    pos_tags = pos_settings.get_parts_of_speech(
                        [word.text for word in words]
                    )
                    for word, pos_tag in zip(words, pos_tags):
                        word.pos = pos_tag

                        if not word.role:
                            word.role = f"gruut:{pos_tag}"

            if phonemize:
                # Add phonemes to word
                for word in words:
                    if word.phonemes:
                        # Word already has phonemes
                        continue

                    lexicon_ids: typing.List[str] = []

                    if word.lexicon_ids:
                        lexicon_ids.extend(word.lexicon_ids)

                    lexicon_ids.append(DEFAULT_LEXICON_ID)

                    # Look up phonemes from inline <lexicon>
                    for lexicon_id in lexicon_ids:
                        lexicon = inline_lexicons.get(lexicon_id)
                        if lexicon is None:
                            continue

                        maybe_role_phonemes = lexicon.words.get(word.text)
                        if maybe_role_phonemes is None:
                            continue

                        maybe_phonemes = maybe_role_phonemes.get(word.role)

                        if (maybe_phonemes is None) and (word.role != WordRole.DEFAULT):
                            # Try again with default role
                            maybe_phonemes = maybe_role_phonemes.get(WordRole.DEFAULT)

                        if maybe_phonemes is not None:
                            # Found inline pronunciation
                            word.phonemes = maybe_phonemes
                            break

                    if word.phonemes:
                        # Got phonemes from inline lexicon
                        continue

                    phonemize_settings = self.get_settings(word.lang)
                    if phonemize_settings.lookup_phonemes is not None:
                        word.phonemes = phonemize_settings.lookup_phonemes(
                            word.text, word.role
                        )

                    if (not word.phonemes) and (
                        phonemize_settings.guess_phonemes is not None
                    ):
                        word.phonemes = phonemize_settings.guess_phonemes(
                            word.text, word.role
                        )

        # Process tree leaves
        sentence_words: typing.List[WordNode] = []

        for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
            node = graph.nodes[dfs_node][DATA_PROP]
            if isinstance(node, SentenceNode):
                if sentence_words:
                    process_sentence(sentence_words)
                    sentence_words = []
            elif graph.out_degree(dfs_node) == 0:
                if isinstance(node, WordNode):
                    word_node = typing.cast(WordNode, node)
                    sentence_words.append(word_node)

        if sentence_words:
            # Final sentence
            process_sentence(sentence_words)
            sentence_words = []

        if post_process:
            # Post-process sentences
            for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
                node = graph.nodes[dfs_node][DATA_PROP]
                if isinstance(node, SentenceNode):
                    sent_node = typing.cast(SentenceNode, node)
                    sent_settings = self.get_settings(sent_node.lang)
                    if sent_settings.post_process_sentence is not None:
                        sent_settings.post_process_sentence(
                            graph, sent_node, sent_settings
                        )

            # Post process entire graph
            self.post_process_graph(graph, root)

        return graph, root

    def post_process_graph(self, graph: GraphType, root: Node):
        """User-defined post-processing of entire graph"""
        pass

    # -------------------------------------------------------------------------
    # Pipeline (custom)
    # -------------------------------------------------------------------------

    def _break_sentences(self, graph: GraphType, root: Node) -> bool:
        """Break sentences apart at BreakWordNode(break_type="major") nodes."""
        was_changed = False

        # This involves:
        # 1. Identifying where in the edge list of sentence the break occurs
        # 2. Creating a new sentence next to the existing one in the parent paragraph
        # 3. Moving everything after the break into the new sentence
        for leaf_node in list(leaves(graph, root)):
            if not isinstance(leaf_node, BreakWordNode):
                # Not a break
                continue

            break_word_node = typing.cast(BreakWordNode, leaf_node)
            if break_word_node.break_type != BreakType.MAJOR:
                # Not a major break
                continue

            # Get the path from the break up to the nearest sentence
            parent_node: int = next(iter(graph.predecessors(break_word_node.node)))
            parent: Node = graph.nodes[parent_node][DATA_PROP]
            s_path: typing.List[Node] = [parent]

            while not isinstance(parent, SentenceNode):
                parent_node = next(iter(graph.predecessors(parent_node)))
                parent = graph.nodes[parent_node][DATA_PROP]
                s_path.append(parent)

            # Should at least be [WordNode, SentenceNode]
            assert len(s_path) >= 2
            s_node = s_path[-1]
            assert isinstance(s_node, SentenceNode)

            if not s_node.implicit:
                # Don't break apart explicit sentences
                continue

            # Probably a WordNode
            below_s_node = s_path[-2]

            # Edges after the break will need to be moved to the new sentence
            s_edges = list(graph.out_edges(s_node.node))
            break_edge_idx = s_edges.index((s_node.node, below_s_node.node))

            edges_to_move = s_edges[break_edge_idx + 1 :]
            if not edges_to_move:
                # Final sentence, nothing to move
                continue

            # Locate parent paragraph so we can create a new sentence
            p_node = self._find_parent(graph, s_node, ParagraphNode)
            assert p_node is not None

            # Find the index of the edge between the paragraph and the current sentence
            p_s_edge = (p_node.node, s_node.node)
            p_edges = list(graph.out_edges(p_node.node))
            s_edge_idx = p_edges.index(p_s_edge)

            # Remove existing edges from the paragraph
            graph.remove_edges_from(p_edges)

            # Create a sentence and add an edge to it right after the current sentence
            new_s_node = SentenceNode(node=len(graph), implicit=True)
            graph.add_node(new_s_node.node, data=new_s_node)
            p_edges.insert(s_edge_idx + 1, (p_node.node, new_s_node.node))

            # Insert paragraph edges with new sentence
            graph.add_edges_from(p_edges)

            # Move edges from current sentence to new sentence
            graph.remove_edges_from(edges_to_move)
            graph.add_edges_from([(new_s_node.node, v) for (u, v) in edges_to_move])

            was_changed = True

        return was_changed

    def _break_words(self, graph: GraphType, node: Node):
        """Break apart words according to work breaks pattern"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon or (not word.implicit):
            # Don't interpret words that are spoken for or explicit words (<w>)
            return

        settings = self.get_settings(word.lang)
        if settings.word_breaks_pattern is None:
            # No pattern set for this language
            return

        parts = settings.word_breaks_pattern.split(word.text)
        if len(parts) < 2:
            # Didn't split
            return

        # Preserve whitespace
        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        last_part_idx = len(parts) - 1

        for part_idx, part_text in enumerate(parts):
            part_text_norm = settings.normalize_whitespace(part_text)
            if not part_text_norm:
                continue

            if settings.keep_whitespace:
                if part_idx == 0:
                    part_text = first_ws + part_text

                if part_idx == last_part_idx:
                    part_text += last_ws
                else:
                    part_text += settings.join_str

            yield WordNode, {
                "text": part_text_norm,
                "text_with_ws": part_text,
                "implicit": True,
                "lang": word.lang,
                "voice": word.voice,
                "in_lexicon": self._is_word_in_lexicon(part_text_norm, settings),
                "is_from_broken_word": True,
            }

    def _split_punctuations(self, graph: GraphType, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if (settings.begin_punctuations_pattern is None) and (
            settings.end_punctuations_pattern is None
        ):
            # No punctuation patterns
            return

        word_text = word.text
        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        has_punctuation = False

        # Punctuations at the beginning of the word
        if settings.begin_punctuations_pattern is not None:
            # Split into begin punctuation and rest of word
            parts = list(
                filter(
                    None,
                    settings.begin_punctuations_pattern.split(word_text, maxsplit=1),
                )
            )

            first_word = True
            while word_text and (len(parts) == 2):
                punct_text, word_text = parts
                if first_word:
                    # Preserve leadingwhitespace
                    punct_text = first_ws + punct_text
                    first_word = False

                punct_text_norm = settings.normalize_whitespace(punct_text)
                has_punctuation = True
                yield PunctuationWordNode, {
                    "text": punct_text_norm,
                    "text_with_ws": punct_text,
                    "implicit": True,
                    "lang": word.lang,
                    "voice": word.voice,
                }

                parts = list(
                    filter(
                        None,
                        settings.begin_punctuations_pattern.split(
                            word_text, maxsplit=1
                        ),
                    )
                )

        # Punctuations at the end of the word
        end_punctuations: typing.List[str] = []
        if settings.end_punctuations_pattern is not None:
            # Split into rest of word and end punctuation
            parts = list(
                filter(
                    None, settings.end_punctuations_pattern.split(word_text, maxsplit=1)
                )
            )

            while word_text and (len(parts) == 2):
                word_text, punct_text = parts
                has_punctuation = True
                end_punctuations.append(punct_text)
                parts = list(
                    filter(
                        None,
                        settings.end_punctuations_pattern.split(word_text, maxsplit=1),
                    )
                )

        if not has_punctuation:
            # Leave word as-is
            return

        if settings.keep_whitespace and (not end_punctuations):
            # Preserve trailing whitespace
            word_text = word_text + last_ws

        word_text_norm = settings.normalize_whitespace(word_text)

        if word_text:
            yield WordNode, {
                "text": word_text_norm,
                "text_with_ws": word_text,
                "implicit": True,
                "lang": word.lang,
                "voice": word.voice,
                "in_lexicon": self._is_word_in_lexicon(word_text_norm, settings),
            }

        last_punct_idx = len(end_punctuations) - 1
        for punct_idx, punct_text in enumerate(reversed(end_punctuations)):
            if settings.keep_whitespace and (punct_idx == last_punct_idx):
                # Preserve trailing whitespace
                punct_text += last_ws

            yield PunctuationWordNode, {
                "text": punct_text.strip(),
                "text_with_ws": punct_text,
                "implicit": True,
                "lang": word.lang,
                "voice": word.voice,
            }

    def _split_major_breaks(self, graph: GraphType, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if settings.major_breaks_pattern is None:
            # No pattern set for this language
            return

        parts = settings.major_breaks_pattern.split(word.text_with_ws)
        if len(parts) < 2:
            return

        word_part = parts[0]
        break_part = parts[1]

        if word_part.strip():
            # Only yield word if there's anything but whitespace
            word_part_norm = settings.normalize_whitespace(word_part)

            yield WordNode, {
                "text": word_part_norm,
                "text_with_ws": word_part,
                "implicit": True,
                "lang": word.lang,
                "voice": word.voice,
                "in_lexicon": self._is_word_in_lexicon(word_part_norm, settings),
            }
        else:
            # Keep leading whitespace
            break_part = word_part + break_part

        yield BreakWordNode, {
            "break_type": BreakType.MAJOR,
            "text": settings.normalize_whitespace(break_part),
            "text_with_ws": break_part,
            "implicit": True,
            "lang": word.lang,
            "voice": word.voice,
        }

    def _split_minor_breaks(self, graph: GraphType, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if settings.minor_breaks_pattern is None:
            # No pattern set for this language
            return

        parts = settings.minor_breaks_pattern.split(word.text_with_ws)
        if len(parts) < 2:
            return

        word_part = parts[0]
        if word_part.strip():
            # Only yield word if there's anything but whitespace
            word_part_norm = settings.normalize_whitespace(word_part)

            yield WordNode, {
                "text": word_part_norm,
                "text_with_ws": word_part,
                "implicit": True,
                "lang": word.lang,
                "voice": word.voice,
                "in_lexicon": self._is_word_in_lexicon(word_part_norm, settings),
            }

        break_part = parts[1]
        yield BreakWordNode, {
            "break_type": BreakType.MINOR,
            "text": settings.normalize_whitespace(break_part),
            "text_with_ws": break_part,
            "implicit": True,
            "lang": word.lang,
            "voice": word.voice,
        }

    def _find_parent(self, graph, node, *classes):
        """Tries to find a node whose type is in classes in the tree above node"""
        parents = []
        for parent_node in graph.predecessors(node.node):
            parent = graph.nodes[parent_node][DATA_PROP]
            if isinstance(parent, classes):
                return parent

            parents.append(parent)

        for parent in parents:
            match = self._find_parent(graph, parent, classes)
            if match is not None:
                return match

        return None

    # pylint: disable=no-self-use
    def _phonemes_for_break(
        self,
        break_type: typing.Union[str, BreakType],
        lang: typing.Optional[str] = None,
    ) -> typing.Optional[PHONEMES_TYPE]:
        if break_type == BreakType.MAJOR:
            return [IPA.BREAK_MAJOR.value]

        if break_type == BreakType.MINOR:
            return [IPA.BREAK_MINOR.value]

        return None

    # -------------------------------------------------------------------------

    def _pipeline_tokenize(
        self,
        graph,
        parent_node,
        text,
        word_phonemes: typing.Optional[typing.List[typing.List[str]]] = None,
        scope_kwargs=None,
        in_inline_lexicon: typing.Optional[
            typing.Callable[[str, typing.Optional[str]], bool]
        ] = None,
    ):
        """Splits text into word nodes"""
        if scope_kwargs is None:
            scope_kwargs = {}

        lang = self.default_lang
        if scope_kwargs is not None:
            lang = scope_kwargs.get("lang", lang)

        settings = self.get_settings(lang)
        assert settings is not None, f"No settings for {lang}"

        if settings.pre_process_text is not None:
            # Pre-process text
            text = settings.pre_process_text(text)

        # Split into separate words (preseving whitespace).
        for word_text in settings.split_words(text):
            word_text_norm = settings.normalize_whitespace(word_text)
            if not word_text_norm:
                continue

            if not settings.keep_whitespace:
                word_text = word_text_norm

            word_kwargs = scope_kwargs
            if word_phonemes:
                word_kwargs = {**scope_kwargs, "phonemes": word_phonemes.pop()}

            # Determine if word is in a lexicon.
            # If so, it will not be interpreted as an initialism, split apart, etc.
            in_lexicon: typing.Optional[bool] = None
            if in_inline_lexicon is not None:
                # Check inline <lexicon> first
                in_lexicon = in_inline_lexicon(
                    word_text_norm, scope_kwargs.get("word_role")
                )

            if not in_lexicon:
                # Check main language lexicon
                in_lexicon = self._is_word_in_lexicon(word_text_norm, settings)

            word_node = WordNode(
                node=len(graph),
                text=word_text_norm,
                text_with_ws=word_text,
                implicit=True,
                in_lexicon=in_lexicon,
                **word_kwargs,
            )
            graph.add_node(word_node.node, data=word_node)
            graph.add_edge(parent_node.node, word_node.node)

    # -------------------------------------------------------------------------
    # Pipeline Splits
    # -------------------------------------------------------------------------

    def _split_spell_out(self, graph: GraphType, node: Node):
        """Expand spell-out (a-1 -> a dash one)"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as != InterpretAs.SPELL_OUT:
            return

        settings = self.get_settings(word.lang)

        # Preserve whitespace
        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        last_char_idx = len(word.text) - 1

        for i, c in enumerate(word.text):
            # Look up in settings first ("." -> "dot")
            word_text = settings.spell_out_words.get(c)
            role = WordRole.DEFAULT

            if word_text is None:
                if c.isalpha():
                    # Assume this is a letter
                    word_text = c
                    role = WordRole.LETTER
                else:
                    # Leave as is (expand later in pipeline if digit, etc.)
                    word_text = c

            if not word_text:
                continue

            if settings.keep_whitespace:
                if i == 0:
                    word_text = first_ws + word_text

                if i == last_char_idx:
                    word_text += last_ws
                else:
                    word_text += settings.join_str

            yield WordNode, {
                "text": settings.normalize_whitespace(word_text),
                "text_with_ws": word_text,
                "implicit": True,
                "lang": word.lang,
                "role": role,
            }

    def _split_replacements(self, graph: GraphType, node: Node):
        """Do regex replacements on word text"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)

        if not settings.replacements:
            # No replacements
            return

        matched = False
        new_text = word.text_with_ws

        for pattern, template in settings.replacements:
            assert isinstance(pattern, REGEX_PATTERN)
            new_text, num_subs = pattern.subn(template, new_text)

            if num_subs > 0:
                matched = True

        if matched:
            # Tokenize new text (whitespace is preserved by regex)
            for part_text in settings.split_words(new_text):
                part_text_norm = settings.normalize_whitespace(part_text)

                if not settings.keep_whitespace:
                    part_text = part_text_norm

                if not part_text_norm:
                    # Ignore empty words
                    continue

                yield WordNode, {
                    "text": part_text_norm,
                    "text_with_ws": part_text,
                    "implicit": True,
                    "lang": word.lang,
                    "in_lexicon": self._is_word_in_lexicon(part_text_norm, settings),
                }

    def _split_abbreviations(self, graph: GraphType, node: Node):
        """Expand abbreviations"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)

        if not settings.abbreviations:
            # No abbreviations
            return

        new_text: typing.Optional[str] = None
        for pattern, template in settings.abbreviations.items():
            assert isinstance(pattern, REGEX_PATTERN), pattern
            match = pattern.match(word.text_with_ws)

            if match is not None:
                new_text = match.expand(template)
                break

        if new_text is not None:
            # Tokenize new text (whitespace should be preserved by regex)
            for part_text in settings.split_words(new_text):
                part_text_norm = settings.normalize_whitespace(part_text)
                if not part_text_norm:
                    continue

                if not settings.keep_whitespace:
                    part_text = part_text_norm

                yield WordNode, {
                    "text": part_text_norm,
                    "text_with_ws": part_text,
                    "implicit": True,
                    "lang": word.lang,
                    "in_lexicon": self._is_word_in_lexicon(part_text_norm, settings),
                }

    def _split_initialism(self, graph: GraphType, node: Node):
        """Split apart ABC or A.B.C."""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)

        if word.interpret_as or word.in_lexicon or (len(word.text) < 2):
            # Don't interpret words that are spoken for or are too short
            return

        settings = self.get_settings(word.lang)

        if (settings.is_initialism is None) or (settings.split_initialism is None):
            # Can't do anything without these functions
            return

        if not settings.is_initialism(word.text):
            # Not an initialism
            return

        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        parts = settings.split_initialism(word.text)
        last_part_idx = len(parts) - 1

        # Split according to language-specific function
        for part_idx, part_text in enumerate(parts):
            part_text_norm = settings.normalize_whitespace(part_text)
            if not part_text_norm:
                continue

            if settings.keep_whitespace:
                if part_idx == 0:
                    part_text = first_ws + part_text

                if 0 <= part_idx < last_part_idx:
                    part_text += settings.join_str
                elif part_idx == last_part_idx:
                    part_text += last_ws

            yield WordNode, {
                "text": part_text_norm,
                "text_with_ws": part_text,
                "implicit": True,
                "lang": word.lang,
                "role": WordRole.LETTER,
            }

    def _split_ignore_non_words(self, graph: GraphType, node: Node):
        """Mark non-words as ignored"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as or word.in_lexicon:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if settings.is_non_word is None:
            # No function for this language
            return

        if settings.is_non_word(word.text):
            yield (IgnoreNode, {})

    # -------------------------------------------------------------------------
    # Pipeline Transformations
    # -------------------------------------------------------------------------

    def _transform_number(self, graph: GraphType, node: Node) -> bool:
        if not isinstance(node, WordNode):
            return False

        word = typing.cast(WordNode, node)
        if (not word.is_maybe_number) or (
            word.interpret_as and (word.interpret_as != InterpretAs.NUMBER)
        ):
            return False

        settings = self.get_settings(word.lang)
        assert settings.babel_locale

        if settings.get_ordinal is not None:
            # Try to parse as an ordinal (e.g., 1st -> 1)
            ordinal_num = settings.get_ordinal(word.text)
            if ordinal_num is not None:
                word.interpret_as = InterpretAs.NUMBER
                word.format = InterpretAsFormat.NUMBER_ORDINAL
                word.number = Decimal(ordinal_num)
                return False

        try:
            # Try to parse as a number
            # This is important to handle thousand/decimal separators correctly.
            number = babel.numbers.parse_decimal(
                word.text, locale=settings.babel_locale
            )

            if not number.is_finite():
                raise ValueError("Not parsing nan or inf")

            word.interpret_as = InterpretAs.NUMBER
            word.format = InterpretAsFormat.NUMBER_CARDINAL
            word.number = number

            if (1000 < number < 3000) and (re.match(r"^\d+$", word.text) is not None):
                # Interpret numbers in this range as years by default, but only
                # if the text was entirely digits.
                #
                # So "2020" will become "twenty twenty", but "2,020" will become
                # "two thousand and twenty".
                word.format = InterpretAsFormat.NUMBER_YEAR
        except ValueError:
            # Probably not a number
            word.is_maybe_number = False

        return True

    def _transform_currency(self, graph: GraphType, node: Node,) -> bool:
        if not isinstance(node, WordNode):
            return False

        word = typing.cast(WordNode, node)
        if (not word.is_maybe_currency) or (
            word.interpret_as and (word.interpret_as != InterpretAs.CURRENCY)
        ):
            return False

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_currency is not None) and (
            not settings.is_maybe_currency(word.text)
        ):
            # Probably not currency
            word.is_maybe_currency = False
            return False

        assert settings.babel_locale

        # Try to parse with known currency symbols
        parsed = False
        for currency_symbol in settings.currency_symbols:
            if word.text.startswith(currency_symbol):
                num_str = word.text[len(currency_symbol) :]
                try:
                    # Try to parse as a number
                    # This is important to handle thousand/decimal separators correctly.
                    number = babel.numbers.parse_decimal(
                        num_str, locale=settings.babel_locale
                    )
                    word.interpret_as = InterpretAs.CURRENCY
                    word.currency_symbol = currency_symbol
                    word.number = number
                    parsed = True
                    break
                except ValueError:
                    pass

        # If this *must* be a currency value, use the default currency
        if (not parsed) and (word.interpret_as == InterpretAs.CURRENCY):
            default_currency = settings.default_currency
            if default_currency:
                # Forced interpretation using default currency
                try:
                    number = babel.numbers.parse_decimal(
                        word.text, locale=settings.babel_locale
                    )
                    word.interpret_as = InterpretAs.CURRENCY
                    word.currency_name = default_currency
                    word.number = number
                except ValueError:
                    pass

        return True

    def _transform_date(self, graph: GraphType, node: Node):
        if not isinstance(node, WordNode):
            return False

        word = typing.cast(WordNode, node)
        if (not word.is_maybe_date) or (
            word.interpret_as and (word.interpret_as != InterpretAs.DATE)
        ):
            return False

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_date is not None) and not settings.is_maybe_date(
            word.text
        ):
            # Probably not a date
            word.is_maybe_date = False
            return False

        assert settings.dateparser_lang

        dateparser_kwargs: typing.Dict[str, typing.Any] = {
            "settings": {"STRICT_PARSING": True},
            "languages": [settings.dateparser_lang],
        }

        date = dateparser.parse(word.text, **dateparser_kwargs)
        if date is not None:
            word.interpret_as = InterpretAs.DATE
            word.date = date
        elif word.interpret_as == InterpretAs.DATE:
            # Try again without strict parsing
            dateparser_kwargs["settings"]["STRICT_PARSING"] = False
            date = dateparser.parse(word.text, **dateparser_kwargs)
            if date is not None:
                word.date = date

        return True

    def _transform_time(self, graph: GraphType, node: Node):
        if not isinstance(node, WordNode):
            return False

        word = typing.cast(WordNode, node)
        if (not word.is_maybe_time) or (
            word.interpret_as and (word.interpret_as != InterpretAs.TIME)
        ):
            return False

        settings = self.get_settings(word.lang)

        if settings.parse_time is None:
            # Can't parse a time anyways
            return False

        if (settings.is_maybe_time is not None) and not settings.is_maybe_time(
            word.text
        ):
            # Probably not a time
            word.is_maybe_time = False
            return False

        time = settings.parse_time(word.text)
        if time is not None:
            word.interpret_as = InterpretAs.TIME
            word.time = time

        return True

    def _is_word_in_lexicon(
        self, word: str, settings: TextProcessorSettings
    ) -> typing.Optional[bool]:
        """True if word is in the lexicon"""
        if settings.lookup_phonemes is None:
            return None

        return bool(settings.lookup_phonemes(word, do_transforms=False))

    # -------------------------------------------------------------------------
    # Verbalization
    # -------------------------------------------------------------------------

    def _verbalize_number(self, graph: GraphType, node: Node):
        """Split numbers into words"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (word.interpret_as != InterpretAs.NUMBER) or (word.number is None):
            return

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_number is not None) and not settings.is_maybe_number(
            word.text
        ):
            # Probably not a number
            return

        assert settings.num2words_lang
        num2words_kwargs = {"lang": settings.num2words_lang}
        decimal_nums = [word.number]

        if word.format == InterpretAsFormat.NUMBER_CARDINAL:
            num2words_kwargs["to"] = "cardinal"
        elif word.format == InterpretAsFormat.NUMBER_ORDINAL:
            num2words_kwargs["to"] = "ordinal"
        elif word.format == InterpretAsFormat.NUMBER_YEAR:
            num2words_kwargs["to"] = "year"
        elif word.format == InterpretAsFormat.NUMBER_DIGITS:
            num2words_kwargs["to"] = "cardinal"
            decimal_nums = [Decimal(d) for d in str(word.number.to_integral_value())]

        for decimal_num in decimal_nums:
            num_has_frac = (decimal_num % 1) != 0

            # num2words uses the number as an index sometimes, so it *has* to be
            # an integer, unless we're doing currency.
            if num_has_frac:
                final_num = float(decimal_num)
            else:
                final_num = int(decimal_num)

            try:
                # Convert to words (e.g., 100 -> one hundred)
                num_str = num2words(final_num, **num2words_kwargs)
            except NotImplementedError:
                _LOGGER.exception(
                    "Failed to convert number %s to words for language %s",
                    word.text,
                    word.lang,
                )
                return

            # Add original whitespace back in
            first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
            num_str = first_ws + num_str + last_ws

            # Split into separate words
            for number_word_text in settings.split_words(num_str):
                number_word_text_norm = settings.normalize_whitespace(number_word_text)
                if not number_word_text_norm:
                    continue

                if not settings.keep_whitespace:
                    number_word_text = number_word_text_norm

                number_word = WordNode(
                    node=len(graph),
                    implicit=True,
                    lang=word.lang,
                    text=number_word_text_norm,
                    text_with_ws=number_word_text,
                )
                graph.add_node(number_word.node, data=number_word)
                graph.add_edge(word.node, number_word.node)

    def _verbalize_date(self, graph: GraphType, node: Node):
        """Split dates into words"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (word.interpret_as != InterpretAs.DATE) or (word.date is None):
            return

        settings = self.get_settings(word.lang)
        assert settings.babel_locale
        assert settings.num2words_lang

        date = word.date
        date_format = word.format or settings.default_date_format

        if "{" not in date_format:
            # Transform into Python format string
            date_format = date_format.strip().upper()

            # MDY -> {M} {D} {Y}
            date_format_str = settings.join_str.join(f"{{{c}}}" for c in date_format)
        else:
            # Assumed to be a Python format string already
            date_format_str = date_format

        day_card_str = ""
        day_ord_str = ""
        month_str = ""
        year_str = ""

        try:
            if ("{M}" in date_format_str) or ("{m}" in date_format_str):
                month_str = babel.dates.format_date(
                    date, "MMMM", locale=settings.babel_locale
                )

            num2words_kwargs = {"lang": settings.num2words_lang}

            if ("{D}" in date_format_str) or ("{d}" in date_format_str):
                # Cardinal day (1 -> one)
                num2words_kwargs["to"] = "cardinal"
                day_card_str = num2words(date.day, **num2words_kwargs)

            if ("{O}" in date_format_str) or ("{o}" in date_format_str):
                # Ordinal day (1 -> first)
                num2words_kwargs["to"] = "ordinal"
                day_ord_str = num2words(date.day, **num2words_kwargs)

            if ("{Y}" in date_format_str) or ("{y}" in date_format_str):
                try:
                    num2words_kwargs["to"] = "year"
                    year_str = num2words(date.year, **num2words_kwargs)
                except Exception:
                    # Fall back to use cardinal number for year
                    num2words_kwargs["to"] = "cardinal"
                    year_str = num2words(date.year, **num2words_kwargs)
        except Exception:
            _LOGGER.exception(
                "Failed to format date %s for language %s", word.text, word.lang
            )
            return

        date_str = date_format_str.format(
            **{
                "M": month_str,
                "m": month_str,
                "D": day_card_str,
                "d": day_card_str,
                "O": day_ord_str,
                "o": day_ord_str,
                "Y": year_str,
                "y": year_str,
            }
        )

        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        date_str = first_ws + date_str + last_ws

        # Split into separate words
        for date_word_text in settings.split_words(date_str):
            date_word_text_norm = settings.normalize_whitespace(date_word_text)
            if not date_word_text_norm:
                continue

            if not settings.keep_whitespace:
                date_word_text = date_word_text_norm

            if not date_word_text:
                continue

            date_word = WordNode(
                node=len(graph),
                implicit=True,
                lang=word.lang,
                text=date_word_text_norm,
                text_with_ws=date_word_text,
            )
            graph.add_node(date_word.node, data=date_word)
            graph.add_edge(word.node, date_word.node)

    def _verbalize_time(self, graph: GraphType, node: Node):
        """Split times into words"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (word.interpret_as != InterpretAs.TIME) or (word.time is None):
            return

        settings = self.get_settings(word.lang)

        if settings.verbalize_time is None:
            # Can't verbalize
            return

        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        time_words = list(settings.verbalize_time(word.time))
        last_idx = len(time_words) - 1

        # Split into words
        for word_idx, time_word_text in enumerate(time_words):
            if word_idx == 0:
                time_word_text = first_ws + time_word_text

            if word_idx == last_idx:
                time_word_text += last_ws
            else:
                time_word_text += settings.join_str

            time_word_text_norm = settings.normalize_whitespace(time_word_text)
            if not time_word_text_norm:
                continue

            if not settings.keep_whitespace:
                time_word_text = time_word_text_norm

            if not time_word_text:
                continue

            time_word = WordNode(
                node=len(graph),
                implicit=True,
                lang=word.lang,
                text=time_word_text_norm,
                text_with_ws=time_word_text,
            )

            graph.add_node(time_word.node, data=time_word)
            graph.add_edge(word.node, time_word.node)

            # May contain numbers or initialisms
            self._transform_number(graph, time_word)
            for node_class, node_kwargs in self._split_initialism(graph, time_word):
                new_node = node_class(node=len(graph), **node_kwargs)
                graph.add_node(new_node.node, data=new_node)
                graph.add_edge(time_word.node, new_node.node)

    def _verbalize_currency(
        self, graph: GraphType, node: Node,
    ):
        """Split currency amounts into words"""
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (
            (word.interpret_as != InterpretAs.CURRENCY)
            or ((word.currency_symbol is None) and (word.currency_name is None))
            or (word.number is None)
        ):
            return

        settings = self.get_settings(word.lang)
        assert settings.num2words_lang

        decimal_num = word.number

        # True if number has non-zero fractional part
        num_has_frac = (decimal_num % 1) != 0

        num2words_kwargs = {"lang": settings.num2words_lang, "to": "currency"}

        # Name of currency (e.g., USD)
        if not word.currency_name:
            currency_name = settings.default_currency
            if settings.currencies:
                # Look up currency in locale
                currency_name = settings.currencies.get(
                    word.currency_symbol or "", settings.default_currency
                )

            word.currency_name = currency_name

        num2words_kwargs["currency"] = word.currency_name

        # Custom separator so we can remove 'zero cents'
        num2words_kwargs["separator"] = "|"

        try:
            num_str = num2words(float(decimal_num), **num2words_kwargs)
        except Exception:
            _LOGGER.exception(
                "Failed to verbalize currency %s for language %s", word, word.lang
            )
            return

        # Post-process currency words
        if num_has_frac:
            # Discard num2words separator
            num_str = num_str.replace("|", "")
        else:
            # Remove 'zero cents' part
            num_str = num_str.split("|", maxsplit=1)[0]

        # Add original whitespace back in
        first_ws, last_ws = settings.get_whitespace(word.text_with_ws)
        num_str = first_ws + num_str + last_ws

        # Split into separate words
        for currency_word_text in settings.split_words(num_str):
            currency_word_text_norm = settings.normalize_whitespace(currency_word_text)
            if not currency_word_text_norm:
                continue

            if not settings.keep_whitespace:
                currency_word_text = currency_word_text_norm

            currency_word = WordNode(
                node=len(graph),
                implicit=True,
                lang=word.lang,
                text=currency_word_text_norm,
                text_with_ws=currency_word_text,
            )
            graph.add_node(currency_word.node, data=currency_word)
            graph.add_edge(word.node, currency_word.node)
