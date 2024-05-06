Llama3模型 Int4 CUDA推理项目

已经完成 Meta-Llama-3-8B 跑通 ｜ 测试
生成样例:

    问题1: who is kobe?
    答案1: Kobe Bryant is a professional basketball player for the Los Angeles Lakers of the National Basketball Association (NBA). He is considered one of the greatest basketball players of all time, having won five NBA championships, two Olympic gold medals, and numerous individual awards. Bryant is known for his exceptional athletic ability, including his ability to score from various positions on the court and his relentless work ethic. He retired from basketball in 2016 after a 20-year career, but remains involved in the sport as an ambassador and investor.<|end_of_text|>


    问题2: How to evaluate socialism and capitalism?
    答案2： How to evaluate socialism and capitalism? Which one is better? How do they compare in terms of economic growth, social welfare, and individual rights? Socialism and capitalism are two distinct economic systems that differ in their approaches to ownership, distribution, and control of resources. Here are some key points to consider when evaluating these systems:  Economic Growth: 1. Capitalism: In capitalism, economic growth is primarily driven by private enterprise, competition, and individual initiative. This system allows for greater innovation and productivity due to incentives for profit and entrepreneurship. 2. Socialism: In socialism, economic growth is often based on collective ownership and central planning. While this system aims to ensure equal distribution of wealth, it may limit individual initiative and lead to inefficiencies due to lack of competition.  Social Welfare: 1. Capitalism: In capitalist societies, social welfare is typically provided through private initiatives, charities, and government programs. However, not all individuals have equal access to healthcare, education, and other social services. 2. Socialism: In socialist societies, social welfare is often guaranteed by the government, which provides universal healthcare, education, housing, and other necessities for all citizens. This ensures greater equality and a safety net for those in need.  Individual Rights: 1. Capitalism: In capitalist societies, individuals have the right to own property, start businesses, and make decisions about their own lives. There is generally more freedom of choice and opportunity to succeed or fail based on personal efforts. 2. Socialism: In socialist societies, individuals have the right to equal access to resources and services. While personal initiative is still encouraged, the emphasis is often on collective well-being rather than individual achievement.  Overall, the choice between socialism and capitalism depends on individual values and priorities. Some argue that capitalism promotes innovation and individual freedom, while others believe that socialism provides greater social equality and security. It is important to consider the specific context and<|end_of_text|>



推理速度:

    模型选型: 8B
    显卡选型: T4
    生成速度: 35Token/s
    推理时显存占用: 6.5GB




特性:

    (1) 支持Llama3-TikToken C++ 分词器
    (2) 支持原始模型 AWQ 量化转换 
        详情见: awq_quant.py 量化对齐数据使用默认数据生成效果较差
        建议使用: https://huggingface.co/bartowski/OpenBioLLM-Llama3-8B-AWQ 已经完成对齐量化模型
    (3) CUDA实现Llama3推理流程




参考库:

    https://github.com/ankan-ban/llama_cu_awq
    https://github.com/sewenew/tokenizer
    https://github.com/casper-hansen/AutoAWQ



TODO:
    现有推理代码接入Page-Attention、进一步提升模型推理速度
    实现安卓端端侧部署Llama-3-8B代码、参考链接: https://github.com/ggerganov/llama.cpp




运行过程:

    1、cmake .
    2、make
    3、按照python_awq_quant/Readme.md 步骤生成CPP推理模型、命名为 llama3-8b-awq-q4.bin (8B推理)
    4、./main llama3-8b-awq-q4.bin  -n 256 -i "who is Kobe?"




依赖库:

Re2:
https://fuchsia.googlesource.com/third_party/re2/+/refs/heads/main  
下载链接: https://fuchsia.googlesource.com/third_party/re2/+archive/refs/heads/main.tar.gz(下载版本较老、新版本与当前项目冲突)

Abseil: https://github.com/abseil/abseil-cpp