import re


def get_gsm8k_final_answer(response):
    return response.split("Final Answer: ")[-1].split("#### ")[-1]


def clean_gsm8k_final_answer(final_answer : str):
    parse_error = 0
    final_answer = final_answer.split("=")[-1]
    # Extract number from final answer
    integer_final_answer = re.findall(r"\d[\d,\.]*", final_answer)
    if len(integer_final_answer) > 1:
        print("MULTIPLE INTEGER FINAL ANSWERS FOUND!!!")
        print(final_answer)
        print(integer_final_answer)
        print("----------------")
        parse_error += 1
    elif len(integer_final_answer) == 0:
        print("NO INTEGER FINAL ANSWERS FOUND!!!")
        print(final_answer)
        print(integer_final_answer)
        print("----------------")
        parse_error = 1
        return "", parse_error
    integer_final_answer = integer_final_answer[0]
    integer_final_answer = integer_final_answer.replace("$", "").replace(",", "")
    return integer_final_answer, parse_error