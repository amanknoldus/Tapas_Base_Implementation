import ast


def parse_answer_text(answer_text):
    try:
        answer = []
        for value in ast.literal_eval(answer_text):
            answer.append(value)
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_text)

    return answer


def parse_coordinates(data_cord):
    answer_coordinates = ast.literal_eval(str(data_cord))
    return answer_coordinates
