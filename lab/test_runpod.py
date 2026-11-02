import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rent

IMAGE = 'ghcr.io/test/engine@sha256:' + 'a' * 64
NO_STOCK = ('There are no instances currently available', 'There are no longer any instances available with the requested specifications')


def spec(image=IMAGE, provision=(('asset', '/app/config.yml', 'x'),)):
    return SimpleNamespace(name='lab', image=image, entrypoint='python3', arguments=['main.py'], env={}, port=5000, disk=60,
                           provision=list(provision), gpu_override=None, hardware_id='rtx-5090-32gb',
                           onstart_script=lambda: 'fetch && exec python3 main.py')


class RunPodTest(unittest.TestCase):
    def create(self, s, *answers, cloud='SECURE', exclude=()):
        sent, answers = [], list(answers)

        def call(method, path, body=None, timeout=60):
            sent.append(dict(body))
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        provider = rent.RunPod(argparse.Namespace(cloud=cloud))
        with patch.object(provider, 'call', call), patch.object(rent, 'log'):
            return provider.create(s, set(exclude)), sent

    def test_a_provisioned_launch_starts_through_the_onstart_script(self):
        _, sent = self.create(spec(), {'id': 'p1'})
        self.assertEqual((sent[0]['dockerEntrypoint'], sent[0]['dockerStartCmd']), (['/bin/sh', '-c'], ['fetch && exec python3 main.py']))

    def test_a_launch_with_nothing_to_provision_keeps_its_entrypoint(self):
        _, sent = self.create(spec(provision=()), {'id': 'p1'})
        self.assertEqual((sent[0]['dockerEntrypoint'], sent[0]['dockerStartCmd']), (['python3'], ['main.py']))

    def test_a_cloud_that_stalled_is_not_tried_again(self):
        handle, sent = self.create(spec(), {'id': 'p1'}, cloud='COMMUNITY', exclude={'COMMUNITY'})
        self.assertEqual(([b['cloudType'] for b in sent], handle['cloud']), (['SECURE'], 'SECURE'))

    def test_both_no_stock_messages_move_on_to_the_next_cloud(self):
        for message in NO_STOCK:
            handle, sent = self.create(spec(), SystemExit(f'RunPod POST /pods failed: 500 {message}'), {'id': 'p2'})
            self.assertEqual(([b['cloudType'] for b in sent], handle['id']), (['SECURE', 'COMMUNITY'], 'p2'), message)

    def test_any_other_error_still_stops_the_run(self):
        with self.assertRaisesRegex(SystemExit, '401'):
            self.create(spec(), SystemExit('RunPod POST /pods failed: 401 unauthorized'))

    def test_a_pod_has_started_only_once_it_has_uptime(self):
        provider = rent.RunPod(argparse.Namespace(cloud='SECURE'))
        with patch.object(provider, 'call', return_value={'desiredStatus': 'RUNNING'}):
            for uptime, started in ((12, True), (None, False)):
                with patch.object(provider, 'uptime', return_value=uptime):
                    self.assertEqual(provider.poll({'id': 'p1', 'gpu': 'x'})[0], started)

    def test_the_readiness_probe_sends_a_user_agent(self):
        with patch('urllib.request.urlopen') as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            self.assertEqual(rent.http_probe('https://p1-5000.proxy.runpod.net/v1/models'), 'ok')
        self.assertTrue(urlopen.call_args.args[0].get_header('User-agent'))


if __name__ == '__main__':
    unittest.main()
