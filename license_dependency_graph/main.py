import os
import sys
import json
import xml.etree.ElementTree as ET
import tarfile
import zipfile
import email
from collections import defaultdict
from pkg_resources import parse_version

import requests

PACKAGES_DIR = './packages'
if not os.path.isdir(PACKAGES_DIR):
    os.makedirs(PACKAGES_DIR)


CMP_OPERATORS = ['>=', '<=', '==', '<', '>', '!=']


EMAIL_PARSER = email.parser.HeaderParser()


def _cmp(val_a, val_b, op):
    table = {
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '<': lambda a, b: a < b,
        '>': lambda a, b: a > b,
        '!=': lambda a, b: a != b
    }
    return table[op](val_a, val_b)


def _split_package_name(package_name):
    """
    example:
        >>> _split_package_name('mindsdb<=1.17.1>1.15.1!=1.15.2')
        >>> ('mindsdb', ('<=', '1.17.1'), ('>', '1.15.1'), ('!=', '1.15.2'))
    """
    s = package_name
    L = []
    for i in reversed(range(len(package_name))):
        for op in CMP_OPERATORS:
            if package_name.startswith(op, i):
                version = s[i + len(op):]
                L.append((op, version))
                s = s[:i]
                break
    return (s, *L)


def _iter_pypi(package_name, cache=dict()):
    if package_name not in cache:
        res = requests.get('https://pypi.org/simple/{}'.format(package_name))
        try:
            res.raise_for_status()
        except Exception:
            cache[package_name] = []
        else:
            tree = ET.fromstring(res.text)
            cache[package_name] = list(tree.findall('.//a'))

    for a in cache[package_name]:
        link = a.get('href')
        link = link.split('#')[0].rstrip('/')
        if link.startswith('/simple'):
            link = 'https://pypi.org' + link
        for ext in ['.tar.gz', '.zip', '.whl']:
            if link.endswith(ext):
                version = link.split('-')[1].rstrip(ext)
                yield (ext, version, link)
                break
        else:
            continue


def _get_licenses_and_requirements(url, cache=dict()):
    dist_name = url.split('/')[-1]
    dist_path = os.path.join(PACKAGES_DIR, dist_name)

    if not os.path.isfile(dist_path):
        res = requests.get(url)
        res.raise_for_status()
        with open(dist_path, 'wb') as f:
            f.write(res.content)

    requirements = set()
    licenses = set()

    def read_pkg_info(data):
        pkg_info = EMAIL_PARSER.parsestr(data)
        for k, v in pkg_info.items():
            if k.lower() == 'license':
                licenses.add(v)

    def read_metadata_json(data):
        metadata = json.loads(data)
        for k, v in metadata.items():
            if k.lower() == 'license':
                licenses.add(v)

    def read_requirements(data):
        lines = data.split('\n')
        for line in map(lambda l: l.strip(), lines):
            if len(line) > 0 and line[0].isalpha():
                requirements.add(line)

    if dist_name.endswith('.tar.gz'):
        with tarfile.open(dist_path, 'r:gz') as tar:
            for member in tar.getmembers():
                name = member.name.lower()
            
                if 'pkg-info' in name:
                    pkg_info = tar.extractfile(member).read().decode()
                    read_pkg_info(pkg_info)

                elif 'requires' in name or 'requirements' in name:
                    data = tar.extractfile(member).read().decode()
                    read_requirements(data)

    elif dist_name.endswith('.zip') or dist_name.endswith('.whl'):
        with zipfile.ZipFile(dist_path) as zip_:
            for member in zip_.namelist():
                name = member.split('/')[0].lower()

                if 'metadata.json' in name:
                    metadata_json = zip_.read(member).decode()
                    read_metadata_json(metadata_json)

                elif 'metadata' in name:
                    metadata = zip_.read(member).decode()
                    read_pkg_info(metadata)

                elif 'requires' in name or 'requirements' in name:
                    data = zip_.read(member).decode()
                    read_requirements(data)

    else:
        raise Exception('exptected tar/zip/whl')

    return list(licenses), list(requirements)


def dep_graph(package_name, D, whitelist=None, branch_cache=dict(), visited_chain=list()):
    name, *versions = _split_package_name(package_name)

    # print('[{}]'.format(' -> '.join([*visited_chain, name])))
    try:
        # group by version
        releases = defaultdict(list)
        for ext, version, link in _iter_pypi(name):
            releases[version].append((ext, link))

        # sort by extension priority
        ext_priority = ['.tar.gz', '.zip', '.whl']
        for k in releases:
            releases[k] = sorted(
                releases[k],
                key=lambda x: ext_priority.index(x[0])
            )

    except Exception as e:
        print('Warning:', e)
        return

    # filter out versions that we don't need
    if len(versions) > 0:
        to_drop = []
        for release_v in releases:
            for op, v in versions:
                if _cmp(parse_version(release_v), parse_version(v), op):
                    break
            else:
                # if none of the conditions was satisfied, drop the relese version
                # NOTE: dict.pop right here will raise RuntimeError
                to_drop.append(release_v)

        for release_v in to_drop:
            releases.pop(release_v, None)

    # extract license and requirements from filtered releases
    # and make recursive call call for each requirement
    for release_v, distributions in releases.items():
        # for now use first distribution
        # later maybe try other in case of failure to get
        # required info from the first distribution
        ext, link = distributions[0]

        key = str((name, release_v))

        print(' ' * (len(visited_chain) * 2), end='')
        print('{}=={}'.format(name, release_v), end='')

        if whitelist is not None and name.lower() in whitelist:
            print(' [whitelisted]')
            D[key] = 'whitelisted'
        elif name in visited_chain:
            print(' [circular dependency]')
            D[key] = 'circular dependency'
        else:
            if key in branch_cache:
                print(' [cached branch]')
                D[key] = branch_cache[key]
            else:
                print()
                license_name, requirements = _get_licenses_and_requirements(link)
                D[key] = {'license': license_name, 'requirements': dict()}
                branch_cache[key] = D[key]
                for req in requirements:
                    dep_graph(
                        package_name=req,
                        D=D[key]['requirements'],
                        branch_cache=branch_cache,
                        visited_chain=[*visited_chain, name],
                        whitelist=whitelist
                    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python ./{} package_name'.format(__file__))

    D = dict()
    try:
        dep_graph(sys.argv[1], D)
    except Exception:
        with open('out.json', 'w') as f:
            f.write(json.dumps(D))
        raise
    else:
        with open('out.json', 'w') as f:
            f.write(json.dumps(D))
