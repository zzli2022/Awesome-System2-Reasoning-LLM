import json
import os
from typing import Optional


class PaperInformation:
    def __init__(
        self, paper: str, link: str, venue: str, date: str, label: Optional[str] = None
    ):
        self.paper = paper
        self.link = link
        self.venue = venue
        self.date = date
        self.label = label

    def __hash__(self):
        return self.label

    def __lt__(self, other: "PaperInformation"):
        self_year, self_month = map(int, self.date.split("-"))
        other_year, other_month = map(int, other.date.split("-"))

        if self_year != other_year:
            return self_year > other_year  # Reverse logic to sort DESCENDING
        if self_month != other_month:
            return self_month > other_month  # Reverse logic to sort DESCENDING
        if self.venue != other.venue:
            return self.venue < other.venue  # Sort venues in descending order
        return self.paper < other.paper  # Sort titles in descending order

    def __eq__(self, other: "PaperInformation"):
        return self.label == other.label


class Utility:
    @staticmethod
    def get_paper_information(raw_paper_dict: dict) -> list[PaperInformation]:
        paper_information_list = []
        for key, value in raw_paper_dict.items():
            if isinstance(value, dict):
                paper_information_list.extend(Utility.get_paper_information(value))
            elif isinstance(value, list):
                for raw_paper_information in value:
                    if len(raw_paper_information.keys()) == 1:
                        assert "label" in raw_paper_information.keys()
                    else:
                        paper_label = raw_paper_information.get("label", None)
                        paper_information = PaperInformation(
                            paper=raw_paper_information["paper"],
                            link=raw_paper_information["link"],
                            venue=raw_paper_information["venue"],
                            date=raw_paper_information["date"],
                            label=paper_label,
                        )
                        assert (
                            paper_label is not None
                            or paper_information not in paper_information_list
                        )
                        paper_information_list.append(paper_information)
            else:
                raise TypeError(f"Unexpected type: {type(value)}")
        return paper_information_list

    @staticmethod
    def fill_paper_dict(
        raw_paper_dict: dict, paper_information_list: list[PaperInformation]
    ) -> dict:
        processed_paper_dict = {}
        for key, value in raw_paper_dict.items():
            if isinstance(value, dict):
                processed_paper_dict[key] = Utility.fill_paper_dict(
                    value, paper_information_list
                )
            elif isinstance(value, list):
                processed_paper_dict[key] = []
                for raw_paper_information in value:
                    if (
                        len(raw_paper_information.keys()) == 1
                        or "label" in raw_paper_information.keys()
                    ):
                        paper_label = raw_paper_information["label"]
                        for paper_information in paper_information_list:
                            if paper_information.label == paper_label:
                                break
                        else:
                            raise ValueError(f"Paper label not found: {paper_label}")
                        processed_paper_dict[key].append(paper_information)
                    else:
                        processed_paper_dict[key].append(
                            PaperInformation(
                                paper=raw_paper_information["paper"],
                                link=raw_paper_information["link"],
                                venue=raw_paper_information["venue"],
                                date=raw_paper_information["date"],
                            )
                        )
            else:
                raise TypeError(f"Unexpected type: {type(value)}")
        return processed_paper_dict

    @staticmethod
    def generate_title_with_level(title: str, title_level: int) -> str:
        return f"{'#' * (title_level + 2)} {title}\n"

    @staticmethod
    def generate_readme_table_with_title(
        title: str, title_level: int, paper_information_list: list[PaperInformation]
    ) -> str:
        result_str = Utility.generate_title_with_level(title, title_level)
        result_str += "|Title|Venue|Date|\n"
        result_str += "|:---|:---|:---|\n"
        paper_information_list.sort()
        for paper_information in paper_information_list:
            result_str += (
                f"|[{paper_information.paper}]({paper_information.link})|"
                f"{paper_information.venue}|"
                f"{paper_information.date}|\n"
            )
        return result_str

    @staticmethod
    def generate_all_table(
        paper_dict: dict, topmost_table_level: int, current_table_str: str
    ) -> str:
        for key, value in paper_dict.items():
            if isinstance(value, dict):
                current_table_str += Utility.generate_title_with_level(
                    key, topmost_table_level
                )
                current_table_str = Utility.generate_all_table(
                    value, topmost_table_level + 1, current_table_str
                )
            elif isinstance(value, list):
                current_table_str += Utility.generate_readme_table_with_title(
                    key, topmost_table_level, value
                )
            else:
                raise TypeError(f"Unexpected type: {type(value)}")
        return current_table_str


def main():
    raw_paper_dict = json.load(open("./assets/paper.json", "r"))
    paper_information_list = Utility.get_paper_information(raw_paper_dict)
    processed_paper_dict = Utility.fill_paper_dict(
        raw_paper_dict, paper_information_list
    )
    all_table_str = Utility.generate_all_table(processed_paper_dict, 0, "")
    with open("table.md", "w") as f:
        f.write(all_table_str)


if __name__ == "__main__":
    main()
