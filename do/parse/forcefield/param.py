from do.models.forcefield.vanderwaals import VanderwaalsParam


class ParamParser:
    @staticmethod
    def parse(file_):
        result = {}
        for line in file_:
            if line.startswith("#"):
                continue

            elif line.startswith("NONBonded "):
                _, type_, inter_epsilon, inter_sigma, intra_epsilon, intra_sigma = line.split()

                result[type_] = VanderwaalsParam(float(inter_epsilon), float(inter_sigma),
                                                 float(intra_epsilon), float(intra_sigma))
            elif len(line.strip()) == 0:
                continue
            else:
                raise ValueError("Unparsable param line: {}".format(line))

        return result
