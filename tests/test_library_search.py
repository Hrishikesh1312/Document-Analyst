from __future__ import annotations

import unittest

from document_analyst.services.library_search import filter_library_documents


class LibrarySearchTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"document_name": "Straße.PDF", "source_path": "/work/Reports/Straße.PDF", "status": "indexed"},
            {"document_name": "notes.txt", "source_path": "/work/notes.txt", "status": "failed"},
            {"document_name": "other.pdf", "source_path": "/archive/other.pdf", "status": "removed"},
            {"document_name": "legacy.md", "source_path": "/work/legacy.md"},
        ]

    def test_names_paths_case_whitespace_and_unicode(self):
        self.assertEqual(filter_library_documents(self.docs, "  STRASSE  "), self.docs[:1])
        self.assertEqual(filter_library_documents(self.docs, "reports"), self.docs[:1])
        self.assertEqual(filter_library_documents(self.docs), self.docs)
        self.assertEqual(filter_library_documents(self.docs, "nonexistent"), [])

    def test_filters_combine_and_legacy_status_defaults_to_indexed(self):
        self.assertEqual(filter_library_documents(self.docs, "work", ["indexed"], [".PDF"]), self.docs[:1])
        self.assertEqual(filter_library_documents(self.docs, statuses=["failed"]), self.docs[1:2])
        self.assertEqual(filter_library_documents(self.docs, statuses=["indexed"], file_types=[".md"]), self.docs[3:])
        self.assertEqual(filter_library_documents(self.docs, statuses=["failed"], file_types=[".pdf"]), [])
        self.assertEqual(len(self.docs), 4)
