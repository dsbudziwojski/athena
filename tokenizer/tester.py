from tokenizer import tokenize

dummy_data = [
    (
        "wiki-1",
        "Python is a high-level, interpreted programming language emphasizing readability.",
        "Python uses indentation to define code blocks instead of braces or keywords. Its dynamic typing and automatic memory management make it a popular choice for rapid development. The language’s extensive standard library and large ecosystem of third-party packages support tasks ranging from web development (Django, Flask) to scientific computing (NumPy, SciPy)."
    ),
    (
        "wiki-2",
        "Monty Python was a British surreal comedy group famed for their groundbreaking sketch show.",
        "Formed in 1969, the troupe included Graham Chapman, John Cleese, Terry Gilliam, Eric Idle, Terry Jones, and Michael Palin. Their television series, Monty Python’s Flying Circus, aired on the BBC and introduced innovative, stream-of-consciousness humor. Their films—such as Life of Brian and The Meaning of Life—remain influential in comedy."
    ),
    (
        "wiki-3",
        "The Pythonidae are a family of non-venomous snakes found in Africa, Asia, and Australia.",
        "Pythons kill prey by constriction and exhibit oviparity, laying clutches of eggs. Species range from the ball python, which grows to around 1.5 meters, to the reticulated python, which can exceed 6 meters. Many pythons are arboreal or terrestrial, and several are kept as exotic pets."
    ),
    (
        "wiki-4",
        "Byte Pair Encoding is a subword tokenization algorithm used in NLP to balance vocabulary size and coverage.",
        "BPE begins with a base vocabulary of characters and iteratively merges the most frequent adjacent pairs into new tokens. This yields a fixed-size vocabulary of subwords that avoids out-of-vocabulary issues while keeping sequence lengths manageable. It is used by models like GPT and RoBERTa."
    ),
    (
        "wiki-5",
        "A tokenizer is a component in NLP that splits raw text into meaningful units (tokens) for downstream processing.",
        "Tokenizers may operate at the character, subword, or word level, applying normalization and segmentation rules. Modern tokenizers like Hugging Face’s Tokenizers library support fast, Rust-backed implementations of BPE, WordPiece, and Unigram algorithms. They can also handle special tokens for padding, unknowns, and sequence boundaries."
    ),
    (
        "wiki-6",
        "IPv6 is the most recent version of the Internet Protocol, designed to replace IPv4 with a vastly larger address space.",
        "An IPv6 address is 128 bits long, represented as eight groups of four hexadecimal digits. It introduces features like stateless address autoconfiguration (SLAAC) and built-in IPsec support. Despite its advantages, global IPv6 adoption continues to lag due to legacy infrastructure and compatibility challenges."
    ),
    (
        "wiki-7",
        "The Schrödinger equation is a fundamental partial differential equation describing the quantum behavior of particles.",
        "In its time-dependent form, it governs how the quantum state evolves over time, incorporating kinetic and potential energy terms. Solutions—wavefunctions—can exhibit interference and tunneling effects. Numerical methods such as finite differences and split-step Fourier are used to approximate its solutions in complex systems."
    ),
    (
        "wiki-8",
        "The Linux kernel is the core of the Linux operating system, managing hardware, processes, and system resources.",
        "Released by Linus Torvalds in 1991, it has grown into a monolithic, modular kernel with support for thousands of devices. Key features include preemptive multitasking, virtual memory, and a rich driver ecosystem. Distributions package the kernel with userland tools to form complete OS environments."
    ),
    (
        "wiki-9",
        "Raspberry Pi is a series of small single-board computers developed to promote teaching of basic computer science.",
        "First released in 2012, the Pi has evolved through multiple generations, featuring ARM-based processors, GPIO pins for hardware interfacing, and support for Linux distributions like Raspberry Pi OS. It’s widely used in education, hobbyist projects, and even industrial applications."
    ),
    (
        "wiki-10",
        "VirtualBox is a free and open-source hypervisor for running guest operating systems on a host machine!",
        "Developed by Oracle, it supports Windows, Linux, macOS, and Solaris hosts. Features include snapshotting, shared folders, virtual networking, and seamless mode. It’s popular for testing software across multiple OS environments without dedicated hardware."
    ),
]


only_data = [t[2] for t in dummy_data]
tokenize(only_data, text_type="list", tokenizer="library", model="BPE")
