"""
Data sampler for loading and sampling text paragraphs from files.
"""
from typing import List, Optional, Union, Iterator
from pathlib import Path
import random


RAW_DATA_PATH = (Path(__file__).parent.parent / "data" / "russell").resolve()


class ParagraphIndexError(Exception):
    """Raised when paragraph indexing is invalid."""
    pass


def load_paragraphs(
    file_path: Path,
    paragraph_range: Optional[Union[slice, int]] = None
) -> List[str]:
    """Load paragraphs from a file, optionally selecting a range.

    Paragraphs are split on double newlines.

    Args:
        file_path: Path to the text file.
        paragraph_range: Slice (e.g., 3:10), int (e.g., 5), or None for all.

    Returns:
        List of paragraph strings.
    """
    text = file_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if paragraph_range is None:
        return paragraphs
    elif isinstance(paragraph_range, int):
        return [paragraphs[paragraph_range]]
    elif isinstance(paragraph_range, slice):
        return paragraphs[paragraph_range]
    else:
        raise ValueError("paragraph_range must be None, int, or slice")


class DataSampler:
    """Sample paragraphs from text files in the data directory.

    Loads all .txt files from RAW_DATA_PATH and provides methods
    to retrieve paragraphs by index, chunks, or random sampling.
    """

    def __init__(self):
        self.fps = tuple(RAW_DATA_PATH.glob("*.txt"))
        self.n_paragraphs = {
            fp.name : len(load_paragraphs(fp)) for fp in self.fps
        }

    def _get_fp(self, file_index: int):
        """Get file path by index.

        Args:
            file_index: Index into the list of text files.

        Returns:
            Path object for the requested file.

        Raises:
            ValueError: If file_index is out of bounds.
        """
        if file_index >= len(self.fps):
            raise ValueError(f"Invalid file index; must be less than {len(self.fps)}")
        elif file_index < 0:
            raise ValueError(f"Invalid file index; must be greater than or equal to 0")
        else:
            return self.fps[file_index]

    def get_paragraph_chunk(self, file_index: int, paragraph_range: Optional[Union[slice, int]] = None) -> List[str]:
        """Get paragraphs from a file by index.

        Args:
            file_index: Index of the file to read.
            paragraph_range: Slice, int, or None for all paragraphs.

        Returns:
            List of paragraph strings.
        """
        return load_paragraphs(self._get_fp(file_index), paragraph_range)

    def iter_paragraph_chunks(self, file_index: int, chunk_size: int, step_size: Optional[int] = None) -> Iterator[List[str]]:
        """Iterate over chunks of paragraphs from a file.

        Args:
            file_index: Index of the file to read.
            chunk_size: Number of paragraphs per chunk.
            step_size: Number of paragraphs to advance between chunks.
                      Defaults to chunk_size (non-overlapping chunks).

        Yields:
            Lists of paragraph strings, each of length chunk_size or less.
        """
        if step_size is None:
            step_size = chunk_size

        paragraphs = load_paragraphs(self._get_fp(file_index))
        for i in range(0, len(paragraphs), step_size):
            yield paragraphs[i:i + chunk_size]

    def sample_segment(self, p_length: int):
        """Sample a random segment of consecutive paragraphs.

        Files are weighted by their paragraph count, so longer files
        are more likely to be selected.

        Args:
            p_length: Number of consecutive paragraphs to sample.

        Returns:
            List of paragraph strings.
        """
        fp = random.choices(self.fps,
                            weights=[self.n_paragraphs[f.name] for f in self.fps],
                            k=1)[0]
        p_index = random.randint(0, self.n_paragraphs[fp.name] - p_length)
        return load_paragraphs(fp, slice(p_index, p_index+p_length))


if __name__ == "__main__":
    sampler = DataSampler()
    x = sampler.sample_segment(10)
    for i in x:
        print(i)
