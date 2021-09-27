#!/usr/bin/env python3
"""
Test sentence segmentation against the "Golden Rules":
https://github.com/diasks2/pragmatic_segmenter#the-golden-rules
"""
import unittest

from gruut import sentences


class GoldenRulesTestCase(unittest.TestCase):
    """Test golden rules of sentence segmentation"""

    def test_rule_1(self):
        """Simple period to end sentence"""
        self.assertEqual(
            _get_sentences("Hello World. My name is Jonas."),
            ["Hello World.", "My name is Jonas."],
        )

    def test_rule_2(self):
        """Question mark to end sentence"""
        self.assertEqual(
            _get_sentences("What is your name? My name is Jonas."),
            ["What is your name?", "My name is Jonas."],
        )

    def test_rule_3(self):
        """Exclamation point to end sentence"""
        self.assertEqual(
            _get_sentences("There it is! I found it."), ["There it is!", "I found it."],
        )

    def test_rule_4(self):
        """One letter upper case abbreviations"""
        # NOTE: gruut removes the "." from E
        self.assertEqual(
            _get_sentences("My name is Jonas E. Smith."), ["My name is Jonas E Smith."],
        )

    def test_rule_5(self):
        """One letter lower case abbreviations"""
        # NOTE: gruut removes the "." from p
        self.assertEqual(
            _get_sentences("Please turn to p. 55."), ["Please turn to p 55."],
        )

    def test_rule_6(self):
        """Two letter lower case abbreviations in the middle of a sentence"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("Were Jane and co. at the party?"),
            ["Were Jane and company at the party?"],
        )

    def test_rule_7(self):
        """Two letter upper case abbreviations in the middle of a sentence"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("They closed the deal with Pitt, Briggs & Co. at noon."),
            ["They closed the deal with Pitt, Briggs and Company at noon."],
        )

    # def test_rule_8(self):
    #     """Two letter lower case abbreviations at the end of a sentence"""
    #     self.assertEqual(
    #         _get_sentences("Let's ask Jane and co. They should know."),
    #         ["Let's ask Jane and company. They should know."],
    #     )

    # def test_rule_9(self):
    #     """Two letter upper case abbreviations at the end of a sentence"""
    #     self.assertEqual(
    #         _get_sentences("They closed the deal with Pitt, Briggs & Co. It closed yesterday."),
    #         ["They closed the deal with Pitt, Briggs and Company. It closed yesterday."],
    #     )

    def test_rule_10(self):
        """Two letter (prepositive) abbreviations"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("I can see Mt. Fuji from here."),
            ["I can see Mount Fuji from here."],
        )

    # def test_rule_11(self):
    #     """Two letter (prepositive & postpositive) abbreviations"""
    #     self.assertEqual(
    #         _get_sentences("St. Michael's Church is on 5th st. near the light."),
    #         ["Saint Michael's Church is on 5th street near the light."],
    #     )

    def test_rule_12(self):
        """Possesive two letter abbreviations"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("That is JFK Jr.'s book."), ["That is J F K Junior's book."],
        )

    def test_rule_13(self):
        """Multi-period abbreviations in the middle of a sentence"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("I visited the U.S.A. last year."),
            ["I visited the U S A last year."],
        )

    # 14) Multi-period abbreviations at the end of a sentence
    # I live in the E.U. How about you?
    # ["I live in the E.U.", "How about you?"]

    # 15) U.S. as sentence boundary
    # I live in the U.S. How about you?
    # ["I live in the U.S.", "How about you?"]

    # 16) U.S. as non sentence boundary with next word capitalized
    # I work for the U.S. Government in Virginia.
    # ["I work for the U.S. Government in Virginia."]

    def test_rule_17(self):
        """U.S. as non sentence boundary"""
        # NOTE: gruut expands abbreviations
        self.assertEqual(
            _get_sentences("I have lived in the U.S. for 20 years."),
            ["I have lived in the U S for 20 years."],
        )

    # 18) A.M. / P.M. as non sentence boundary and sentence boundary
    # At 5 a.m. Mr. Smith went to the bank. He left the bank at 6 P.M. Mr. Smith then went to the store.
    # ["At 5 a.m. Mr. Smith went to the bank.", "He left the bank at 6 P.M.", "Mr. Smith then went to the store."]

    def test_rule_19(self):
        """Number as non sentence boundary"""
        self.assertEqual(
            _get_sentences("She has $100.00 in her bag."),
            ["She has $100.00 in her bag."],
        )

    # 21) Parenthetical inside sentence
    # He teaches science (He previously worked for 5 years as an engineer.) at the local University.
    # ["He teaches science (He previously worked for 5 years as an engineer.) at the local University."]

    # 22) Email addresses
    # Her email is Jane.Doe@example.com. I sent her an email.
    # ["Her email is Jane.Doe@example.com.", "I sent her an email."]

    # 23) Web addresses
    # The site is: https://www.example.50.com/new-site/awesome_content.html. Please check it out.
    # ["The site is: https://www.example.50.com/new-site/awesome_content.html.", "Please check it out."]

    # 24) Single quotations inside sentence
    # She turned to him, 'This is great.' she said.
    # ["She turned to him, 'This is great.' she said."]

    # 25) Double quotations inside sentence
    # She turned to him, "This is great." she said.
    # ["She turned to him, \"This is great.\" she said."]

    # 26) Double quotations at the end of a sentence
    # She turned to him, "This is great." She held the book out to show him.
    # ["She turned to him, \"This is great.\"", "She held the book out to show him."]

    def test_rule_27(self):
        """Double punctuation (exclamation point)"""
        self.assertEqual(
            _get_sentences("Hello!! Long time no see."),
            ["Hello!!", "Long time no see."],
        )

    def test_rule_28(self):
        """Double punctuation (question mark)"""
        self.assertEqual(
            _get_sentences("Hello?? Who is there?"), ["Hello??", "Who is there?"],
        )

    def test_rule_29(self):
        """Double punctuation (exclamation point / question mark)"""
        self.assertEqual(
            _get_sentences("Hello!? Is that you?"), ["Hello!?", "Is that you?"],
        )

    def test_rule_30(self):
        """Double punctuation (question mark / exclamation point)"""
        self.assertEqual(
            _get_sentences("Hello?! Is that you?"), ["Hello?!", "Is that you?"],
        )

    # 31) List (period followed by parens and no period to end item)
    # 1.) The first item 2.) The second item
    # ["1.) The first item", "2.) The second item"]

    # 32) List (period followed by parens and period to end item)
    # 1.) The first item. 2.) The second item.
    # ["1.) The first item.", "2.) The second item."]

    # 33) List (parens and no period to end item)
    # 1) The first item 2) The second item
    # ["1) The first item", "2) The second item"]

    # 34) List (parens and period to end item)
    # 1) The first item. 2) The second item.
    # ["1) The first item.", "2) The second item."]

    # 35) List (period to mark list and no period to end item)
    # 1. The first item 2. The second item
    # ["1. The first item", "2. The second item"]

    # 36) List (period to mark list and period to end item)
    # 1. The first item. 2. The second item.
    # ["1. The first item.", "2. The second item."]

    # 37) List with bullet
    # • 9. The first item • 10. The second item
    # ["• 9. The first item", "• 10. The second item"]

    # 38) List with hypthen
    # ⁃9. The first item ⁃10. The second item
    # ["⁃9. The first item", "⁃10. The second item"]

    # 39) Alphabetical list
    # a. The first item b. The second item c. The third list item
    # ["a. The first item", "b. The second item", "c. The third list item"]

    def test_rule_40(self):
        """Errant newlines in the middle of sentences (PDF)"""
        self.assertEqual(
            _get_sentences("This is a sentence\ncut off in the middle because pdf."),
            ["This is a sentence cut off in the middle because pdf."],
        )

    def test_rule_41(self):
        """Errant newlines in the middle of sentences"""
        self.assertEqual(
            _get_sentences("It was a cold \nnight in the city."),
            ["It was a cold night in the city."],
        )

    # 42) Lower case list separated by newline
    # features\ncontact manager\nevents, activities\n
    # ["features", "contact manager", "events, activities"]

    # 43) Geo Coordinates
    # You can find it at N°. 1026.253.553. That is where the treasure is.
    # ["You can find it at N°. 1026.253.553.", "That is where the treasure is."]

    # 44) Named entities with an exclamation point
    # She works at Yahoo! in the accounting department.
    # ["She works at Yahoo! in the accounting department."]

    # 45) I as a sentence boundary and I as an abbreviation
    # We make a good team, you and I. Did you see Albert I. Jones yesterday?
    # ["We make a good team, you and I.", "Did you see Albert I. Jones yesterday?"]

    # 46) Ellipsis at end of quotation
    # Thoreau argues that by simplifying one’s life, “the laws of the universe will appear less complex. . . .”
    # ["Thoreau argues that by simplifying one’s life, “the laws of the universe will appear less complex. . . .”"]

    # 47) Ellipsis with square brackets
    # "Bohr [...] used the analogy of parallel stairways [...]" (Smith 55).
    # ["\"Bohr [...] used the analogy of parallel stairways [...]\" (Smith 55)."]

    # 48) Ellipsis as sentence boundary (standard ellipsis rules)
    # If words are left off at the end of a sentence, and that is all that is omitted, indicate the omission with ellipsis marks (preceded and followed by a space) and then indicate the end of the sentence with a period . . . . Next sentence.
    # ["If words are left off at the end of a sentence, and that is all that is omitted, indicate the omission with ellipsis marks (preceded and followed by a space) and then indicate the end of the sentence with a period . . . .", "Next sentence."]

    # 49) Ellipsis as sentence boundary (non-standard ellipsis rules)
    # I never meant that.... She left the store.
    # ["I never meant that....", "She left the store."]

    # def test_rule_49(self):
    #     """Ellipsis as sentence boundary (non-standard ellipsis rules)"""
    #     self.assertEqual(
    #         _get_sentences("I never meant that.... She left the store."),
    #         ["I never meant that....", "She left the store."],
    #     )

    # 50) Ellipsis as non sentence boundary
    # I wasn’t really ... well, what I mean...see . . . what I'm saying, the thing is . . . I didn’t mean it.
    # ["I wasn’t really ... well, what I mean...see . . . what I'm saying, the thing is . . . I didn’t mean it."]

    # 51) 4-dot ellipsis
    # One further habit which was somewhat weakened . . . was that of combining words into self-interpreting compounds. . . . The practice was not abandoned. . . .
    # ["One further habit which was somewhat weakened . . . was that of combining words into self-interpreting compounds.", ". . . The practice was not abandoned. . . ."]


def _get_sentences(text):
    return [
        s.text
        for s in sentences(text, verbalize_numbers=False, verbalize_currency=False)
    ]


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
