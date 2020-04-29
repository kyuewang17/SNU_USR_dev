import sys
import getopt


def parse_arguments(args):
    FILE_NAME = args[0]
    _IS_ROS_EMBEDDED = None
    _AGENT_TYPE, _AGENT_NAME = None, None

    try:
        methods, etc_args = \
            getopt.getopt(args[1:],
                          "hm:t:n:",
                          ["help", "method=", "agent_type=", "agent_name="])
    except getopt.GetoptError:
        print("[ERROR] Option Input Methods are WRONG!")
        sys.exit(2)

    for method, arg in methods:
        if method in ("-h", "--help"):
            print(FILE_NAME,
                  "-m <(bool)ROS Switch> | -t <(str) Agent Type> | -n <(str) Agent Name>")
            sys.exit()
        elif method in ("-m", "--method"):
            _IS_ROS_EMBEDDED = arg
        elif method in ("-t", "--agent_type"):
            _AGENT_TYPE = arg
        elif method in ("-n", "--agent_name"):
            _AGENT_NAME = arg

    if _AGENT_TYPE not in ["static", "dynamic"]:
        if _AGENT_TYPE is None:
            print("[WARNING!] Undetermined Agent Type!")
        else:
            print("[WARNING!] Agent Type is set as '%s'!" % _AGENT_TYPE)

    return [_IS_ROS_EMBEDDED, _AGENT_TYPE, _AGENT_NAME]


def main(args):
    _INPUT_METHODS = parse_arguments(args)
    print(_INPUT_METHODS)


if __name__ == '__main__':
    main(sys.argv)
