// SAX style JSON parser
// Copyright 2022 Kenta Ida 
// SPDX-License-Identifier: MIT

use core::fmt::Display;

use embedded_io::{blocking::Read, Io};
use heapless::Vec;
use nom::{
    branch::alt,
    bytes::complete::{escaped, is_not, tag},
    character::complete::{char, one_of},
    combinator::{cut, map, peek, value, opt},
    error::{context, ContextError, ErrorKind, ParseError},
    number::complete::recognize_float_or_exceptions,
    sequence::{preceded, terminated, pair},
    IResult,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonNode<'a> {
    StartMap,
    EndMap,
    StartArray,
    EndArray,
    Key(JsonScalarValue<'a>),
    Value(JsonScalarValue<'a>),
}


#[cfg_attr(doc, aquamarine::aquamarine)]
/// ```mermaid
/// stateDiagram-v2
/// Start --> End: EOF
/// Start --> MapStart: '{'
/// Start --> ArrayStart: '['
/// MapStart --> MapKey
/// MapKey --> Pop: '}'
/// MapKey --> KeyDelimiter: (scalar)
/// KeyDelimiter --> MapValue: ':'
/// MapValue --> MapStart: '{'
/// MapValue --> ArrayStart: '['
/// MapValue --> MapPairDelimiter: (scalar)
/// MapPairDelimiter --> Pop: '}'
/// MapPairDelimiter --> MapKey: ','
/// ArrayStart --> ArrayValue
/// ArrayValue --> MapStart: '{'
/// ArrayValue --> ArrayStart: '['
/// ArrayValue --> Pop: ']'
/// ArrayValue --> ArrayValueDelimiter: (scalar)
/// ArrayValueDelimiter --> Pop: ']'
/// ArrayValueDelimiter --> ArrayValue: ','
/// Pop --> (top of state stack)
/// ```
#[derive(Clone, Copy, Debug)]
enum ParserState {
    Start,
    StartContinueString,
    End,
    MapStart,
    MapKey,
    KeyDelimiter,
    MapValue,
    MapValueContinueString,
    MapPairDelimiter,
    ArrayStart,
    ArrayValue,
    ArrayValueContinueString,
    ArrayValueDelimiter,
    Pop,
}

pub struct Parser<const BUFFER_SIZE: usize, const MAX_DEPTH: usize> {
    state: ParserState,
    buffer: Vec<u8, BUFFER_SIZE>,
    state_stack: Vec<ParserState, MAX_DEPTH>,
    bytes_remaining: Option<usize>,
    string_offset: usize,
}

fn from_utf8_possible(bytes: &[u8]) -> (usize, &str) {
    for length in (0..=bytes.len()).rev() {
        if let Ok(s) = core::str::from_utf8(&bytes[..length]) {
            return (length, s);
        }
    }
    (0, "")
}

fn escaped_str<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
    escaped(is_not("\"\\"), '\\', one_of("\"n\\"))(i)
}

fn json_string<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, (&'a str, bool), E> {
    context(
        "string",
        preceded(
            char('\"'), 
            pair(map(opt(escaped_str), |r| r.unwrap_or("")), map(opt(char('\"')), |r| r.is_some()))
        ),
    )(i)
}
fn json_string_continuing<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, (&'a str, bool), E> {
    context(
        "string",
        pair(escaped_str, map(opt(char('\"')), |r| r.is_some())),
    )(i)
}
fn json_null<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (), E> {
    value((), tag("null"))(input)
}
fn json_boolean<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, bool, E> {
    alt((value(true, tag("true")), value(false, tag("false"))))(input)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonScalarValue<'a> {
    Null,
    Boolean(bool),
    String(&'a str),
    ContinuingString(&'a str, usize, bool),
    Number(JsonNumber),
}
impl<'a> Display for JsonScalarValue<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Boolean(v) => write!(f, "{}", v),
            Self::String(v) => write!(f, "\"{}\"", v),
            Self::ContinuingString(v, offset, terminated) => match (offset, terminated) {
                (0, false) => write!(f, "\"{}", v),
                (0, true) => write!(f, "\"{}\"", v),
                (_, false) => write!(f, "{}", v),
                (_, true) => write!(f, "{}\"", v),
            },
            Self::Number(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonNumber {
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}
impl Display for JsonNumber {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::I32(n) => write!(f, "{}", n),
            Self::U32(n) => write!(f, "{}", n),
            Self::I64(n) => write!(f, "{}", n),
            Self::U64(n) => write!(f, "{}", n),
            Self::F32(n) => write!(f, "{}", n),
            Self::F64(n) => write!(f, "{}", n),
        }
    }
}

impl JsonNumber {
    pub fn try_parse(s: &str) -> Result<JsonNumber, ()> {
        s.parse::<i32>()
            .map(|v| JsonNumber::I32(v))
            .or_else(|_| s.parse::<u32>().map(|v| JsonNumber::U32(v)))
            .or_else(|_| s.parse::<i64>().map(|v| JsonNumber::I64(v)))
            .or_else(|_| s.parse::<u64>().map(|v| JsonNumber::U64(v)))
            .or_else(|_| s.parse::<f32>().map(|v| JsonNumber::F32(v)))
            .or_else(|_| s.parse::<f64>().map(|v| JsonNumber::F64(v)))
            .map_err(|_| ())
    }
}

impl Into<f32> for JsonNumber {
    fn into(self) -> f32 {
        match self {
            Self::I32(n) => n as f32,
            Self::U32(n) => n as f32,
            Self::I64(n) => n as f32,
            Self::U64(n) => n as f32,
            Self::F32(n) => n as f32,
            Self::F64(n) => n as f32,
        }
    }
}
impl Into<f64> for JsonNumber {
    fn into(self) -> f64 {
        match self {
            Self::I32(n) => n as f64,
            Self::U32(n) => n as f64,
            Self::I64(n) => n as f64,
            Self::U64(n) => n as f64,
            Self::F32(n) => n as f64,
            Self::F64(n) => n as f64,
        }
    }
}

fn json_scalar_value<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    _is_eof: bool,
) -> impl FnMut(&'a str) -> IResult<&'a str, JsonScalarValue, E> {
    alt((
        map(json_string, |(s, terminated)| if terminated { JsonScalarValue::String(s) } else { JsonScalarValue::ContinuingString(s, 0, false) }),
        map(json_boolean, |b| JsonScalarValue::Boolean(b)),
        map(json_null, |_| JsonScalarValue::Null),
        terminated(
            |i| {
                recognize_float_or_exceptions(i).and_then(|p| {
                    JsonNumber::try_parse(p.1)
                        .map(|n| (p.0, JsonScalarValue::Number(n)))
                        .or(Err(nom::Err::Failure(E::from_error_kind(
                            i,
                            ErrorKind::Float,
                        ))))
                })
            },
            peek(one_of(",]} \t\n\0")),
        ),
    ))
}

#[derive(Debug)]
pub enum ParserError<InputError, CallbackError> {
    InputError(InputError),
    Fail(nom::error::ErrorKind),
    Callback(CallbackError),
    ValueTooLong,
}
impl<InputError, CallbackError> From<InputError> for ParserError<InputError, CallbackError> {
    fn from(err: InputError) -> Self {
        Self::InputError(err)
    }
}

pub enum ParserCallbackAction {
    Nothing,
    End,
}

impl<const BUFFER_SIZE: usize, const MAX_DEPTH: usize> Parser<BUFFER_SIZE, MAX_DEPTH> {
    pub const fn new() -> Self {
        Self {
            state: ParserState::Start,
            buffer: Vec::new(),
            state_stack: Vec::new(),
            bytes_remaining: None,
            string_offset: 0,
        }
    }

    pub fn set_bytes_remaining(&mut self, bytes_remaining: Option<usize>) {
        self.bytes_remaining = bytes_remaining;
    }

    pub fn parse<I: Read, F, CallbackError>(
        &mut self,
        reader: &mut I,
        mut callback: F,
    ) -> Result<bool, ParserError<I::Error, CallbackError>>
    where
        F: for<'node> FnMut(JsonNode<'node>) -> Result<ParserCallbackAction, CallbackError>,
    {
        loop {
            // Fill buffer.
            let bytes_in_buffer = self.buffer.len();
            unsafe {
                let capacity = self.buffer.capacity();
                self.buffer.set_len(capacity);
            }
            let bytes_written = reader.read(&mut self.buffer[bytes_in_buffer..])?;
            unsafe { self.buffer.set_len(bytes_in_buffer + bytes_written) };

            // Update number of bytes remaining in the stream.
            self.bytes_remaining = self
                .bytes_remaining
                .and_then(|bytes_remaining| Some(bytes_remaining - bytes_written));
            // Checks if all data in the stream is read into the buffer or not.
            let is_eof = match self.bytes_remaining {
                Some(0) => true,
                Some(_) => {
                    if !self.buffer.is_full() && bytes_written == 0 {
                        // Some bytes remaining in the stream, but at this time there are no new data.
                        break Ok(false);
                    }
                    false
                },
                _ => bytes_written == 0,    // If the stream length is indeterminate, check number of bytes read to detect end of stream. 
            };
            // Put EOF marker at the last of the buffer if there are no data exists in the stream.
            if is_eof && !self.buffer.is_full() {
                match self.buffer.last() {
                    Some(0) => {}
                    _ => {
                        self.buffer.push(0).unwrap();
                    }
                }
            }

            // Check if currently in the middle of continuing string or not.
            let is_string_continuing = match &self.state {
                ParserState::StartContinueString => true,
                ParserState::MapValueContinueString => true,
                ParserState::ArrayValueContinueString => true,
                _ => false,
            };

            let (_bytes_consumed, input) = from_utf8_possible(&self.buffer);
            // Trim the first whitespaces in the buffer if current state is not the middle of continuing string.
            let trimmed = if is_string_continuing { input } else { input.trim_start() };

            // Trim whitespaces and retry to refill the buffer.
            if input.len() != trimmed.len() {
                // Consume trimmed bytes
                let bytes_consumed = input.as_bytes().len() - trimmed.as_bytes().len();
                let buffer_len = self.buffer.len();
                if bytes_consumed > 0 {
                    // Discard the consumed data from the buffer.
                    self.buffer.copy_within(bytes_consumed..buffer_len, 0);
                    self.buffer.truncate(buffer_len - bytes_consumed);
                }
                break Ok(false);
            }
            let result: IResult<_, _, (&str, ErrorKind)> = match self.state {
                ParserState::Start => alt((
                    // Initial state.
                    value((ParserState::End, None), char('\0')),
                    value((ParserState::MapStart, None), char('{')),
                    value((ParserState::ArrayStart, None), char('[')),
                    map(json_scalar_value(is_eof), |v| {
                        let next_state = match &v {
                            JsonScalarValue::ContinuingString(_, _, _) => ParserState::StartContinueString,
                            _ => ParserState::Start,
                        };
                        (next_state, Some(v))
                    }),
                ))(trimmed),
                ParserState::StartContinueString => map(json_string_continuing, |(s, terminated)| (if terminated { ParserState::Start } else { ParserState::StartContinueString }, Some(JsonScalarValue::ContinuingString(s, self.string_offset, terminated))))(trimmed),
                ParserState::MapStart => Ok((trimmed, (ParserState::MapKey, None))),
                ParserState::MapKey => alt((
                    // Map key
                    value((ParserState::Pop, None), char('}')), // end of map.
                    map(cut(json_scalar_value(is_eof)), |v| {
                        (ParserState::KeyDelimiter, Some(v))
                    }),
                ))(trimmed),
                ParserState::KeyDelimiter =>
                // Map key is processed, waiting key-value delimiter ':'
                {
                    value((ParserState::MapValue, None), char(':'))(trimmed)
                }
                ParserState::MapValue => alt((
                    // Map value
                    value((ParserState::MapStart, None), char('{')),
                    value((ParserState::ArrayStart, None), char('[')),
                    map(json_scalar_value(is_eof), |v| {
                        let next_state = match &v {
                            JsonScalarValue::ContinuingString(_, _, _) => ParserState::MapValueContinueString,
                            _ => ParserState::MapPairDelimiter,
                        };
                        (next_state, Some(v))
                    }),
                ))(trimmed),
                ParserState::MapValueContinueString => map(json_string_continuing, |(s, terminated)| (if terminated { ParserState::MapPairDelimiter } else { ParserState::MapValueContinueString }, Some(JsonScalarValue::ContinuingString(s, self.string_offset, terminated))))(trimmed),
                ParserState::MapPairDelimiter => alt((
                    value((ParserState::Pop, None), char('}')),
                    value((ParserState::MapKey, None), char(',')),
                ))(trimmed),
                ParserState::ArrayStart => Ok((trimmed, (ParserState::ArrayValue, None))),
                ParserState::ArrayValue => alt((
                    value((ParserState::MapStart, None), char('{')),
                    value((ParserState::ArrayStart, None), char('[')),
                    value((ParserState::Pop, None), char(']')),
                    map(json_scalar_value(is_eof), |v| {
                        let next_state = match &v {
                            JsonScalarValue::ContinuingString(_, _, _) => ParserState::ArrayValueContinueString,
                            _ => ParserState::ArrayValueDelimiter,
                        };
                        (next_state, Some(v))
                    }),
                ))(trimmed),
                ParserState::ArrayValueContinueString => map(json_string_continuing, |(s, terminated)| (if terminated { ParserState::ArrayValueDelimiter } else { ParserState::ArrayValueContinueString }, Some(JsonScalarValue::ContinuingString(s, self.string_offset, terminated))))(trimmed),
                ParserState::ArrayValueDelimiter => alt((
                    value((ParserState::Pop, None), char(']')),
                    value((ParserState::ArrayValue, None), char(',')),
                ))(trimmed),
                ParserState::Pop => {
                    let state = self.state_stack.pop().unwrap();
                    Ok((trimmed, (state, None)))
                }
                ParserState::End => {
                    break Ok(true);
                }
            };
            match result {
                Ok((remaining, (new_state, value))) => {
                    let callback_result = match (self.state, new_state) {
                        (state, ParserState::ArrayStart) => {
                            let pop_state = match state {
                                ParserState::ArrayValue => ParserState::ArrayValueDelimiter,
                                ParserState::MapValue => ParserState::MapPairDelimiter,
                                state => state,
                            };
                            self.state_stack.push(pop_state).unwrap();
                            callback(JsonNode::StartArray)
                        }
                        (state, ParserState::MapStart) => {
                            let pop_state = match state {
                                ParserState::ArrayValue => ParserState::ArrayValueDelimiter,
                                ParserState::MapValue => ParserState::MapPairDelimiter,
                                state => state,
                            };
                            self.state_stack.push(pop_state).unwrap();
                            callback(JsonNode::StartMap)
                        }
                        (ParserState::ArrayValue, ParserState::Pop) => callback(JsonNode::EndArray),
                        (ParserState::ArrayValueDelimiter, ParserState::Pop) => {
                            callback(JsonNode::EndArray)
                        }
                        (ParserState::MapKey, ParserState::Pop) => callback(JsonNode::EndMap),
                        (ParserState::MapPairDelimiter, ParserState::Pop) => {
                            callback(JsonNode::EndMap)
                        }
                        (ParserState::MapKey, _) => callback(JsonNode::Key(value.unwrap())),
                        (_, _) => {
                            match value {
                                Some(JsonScalarValue::ContinuingString(string, offset, terminatd)) => {
                                    // If the value is continuing string, store current offset in string.
                                    self.string_offset = offset + string.len();
                                    callback(JsonNode::Value(JsonScalarValue::ContinuingString(string, offset, terminatd)))
                                },
                                Some(value) => callback(JsonNode::Value(value)),
                                _ => Ok(ParserCallbackAction::Nothing),
                            }
                        }
                    };
                    self.state = new_state;
                    let bytes_consumed = input.as_bytes().len() - remaining.as_bytes().len();
                    let buffer_len = self.buffer.len();
                    if bytes_consumed > 0 {
                        // Discard the consumed data from the buffer.
                        self.buffer.copy_within(bytes_consumed..buffer_len, 0);
                        self.buffer.truncate(buffer_len - bytes_consumed);
                    }
                    match callback_result {
                        Ok(ParserCallbackAction::Nothing) => {}
                        Ok(ParserCallbackAction::End) => {
                            self.state = ParserState::End;
                        }
                        Err(err) => {
                            break Err(ParserError::Callback(err));
                        }
                    }
                    continue;
                }
                Err(err) => {
                    match err {
                        nom::Err::Error((_, kind)) => break Err(ParserError::Fail(kind)),
                        nom::Err::Failure((_, kind)) => break Err(ParserError::Fail(kind)),
                        nom::Err::Incomplete(_) => break Err(ParserError::Fail(nom::error::ErrorKind::Eof)),
                    }
                }
            }
        }
    }
}

pub struct BufferReader<'a> {
    buffer: &'a [u8],
    position: usize,
}
impl<'a> BufferReader<'a> {
    pub const fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }
}
impl<'a> Io for BufferReader<'a> {
    type Error = core::convert::Infallible;
}
impl<'a> Read for BufferReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let bytes_remaining = self.buffer.len() - self.position;
        let bytes_to_read = bytes_remaining.min(buf.len());
        buf[..bytes_to_read]
            .copy_from_slice(&self.buffer[self.position..self.position + bytes_to_read]);
        self.position += bytes_to_read;
        Ok(bytes_to_read)
    }
}

pub type DefaultParserCallbackResult = Result<ParserCallbackAction, core::convert::Infallible>;

#[cfg(test)]
mod test {
    use super::*;

    fn setup_parser<const BUFFER_SIZE: usize, const MAX_DEPTH: usize>(
        input: &'static str,
    ) -> (Parser<BUFFER_SIZE, MAX_DEPTH>, BufferReader<'static>) {
        let mut parser = Parser::new();
        parser.set_bytes_remaining(Some(input.as_bytes().len())); // In order to parse single number literal correctly, the parser must know about the position of the EOF to detect end of number.
        (parser, BufferReader::new(input.as_bytes()))
    }
    #[test]
    fn test_parser_empty_string() {
        let (mut parser, mut input) = setup_parser::<20, 4>("    \"\"");
        let mut expected = [JsonNode::Value(JsonScalarValue::String(""))].into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    Result::<_, core::convert::Infallible>::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_single_string() {
        let (mut parser, mut input) = setup_parser::<20, 4>("    \"    hogeほげ\"");
        let mut expected = [JsonNode::Value(JsonScalarValue::String("    hogeほげ"))].into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    Result::<_, core::convert::Infallible>::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_single_continuing_string() {
        let (mut parser, mut input) = setup_parser::<9, 4>("    \"    hogeほげ\"");
        // The parser buffer can stores 9 bytes, so the buffer actually contains "    hoge" and the first byte of "ほ", 
        // but it fails to decode because remaining bytes of "ほ" is not in the buffer.
        // So the parser returns two ContinuingString below.
        let mut expected = [
            JsonNode::Value(JsonScalarValue::ContinuingString("    hoge", 0, false)),
            JsonNode::Value(JsonScalarValue::ContinuingString("ほげ", 8, true)),
        ].into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    Result::<_, core::convert::Infallible>::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_single_null() {
        let (mut parser, mut input) = setup_parser::<128, 4>("  null");
        parser
            .parse(&mut input, |node| match node {
                JsonNode::Value(JsonScalarValue::Null) => {
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                }
                v => panic!("unexpected JSON node - {:?}", v),
            })
            .expect("Parser must not fail.");
    }
    #[test]
    fn test_parser_single_number() {
        let (mut parser, mut input) = setup_parser::<128, 4>("123.0");
        parser
            .parse(&mut input, |node| match node {
                JsonNode::Value(JsonScalarValue::Number(JsonNumber::F32(123.0))) => {
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                }
                v => panic!("unexpected JSON node - {:?}", v),
            })
            .expect("Parser must not fail.");
    }
    #[test]
    fn test_parser_empty_map() {
        let (mut parser, mut input) = setup_parser::<128, 4>("{}");
        let mut expected = [JsonNode::StartMap, JsonNode::EndMap].into_iter();
        parser
            .parse(&mut input, |node| {
                assert_eq!(Some(node), expected.next());
                DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
            })
            .expect("Parser must not fail.");
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_empty_array() {
        let (mut parser, mut input) = setup_parser::<128, 4>("[]");
        let mut expected = [JsonNode::StartArray, JsonNode::EndArray].into_iter();
        parser
            .parse(&mut input, |node| {
                assert_eq!(Some(node), expected.next());
                DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
            })
            .expect("Parser must not fail.");
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }

    #[test]
    fn test_parser_nested_array() {
        let (mut parser, mut input) = setup_parser::<6, 4>(
            "[[  ],
            [[]]]",
        );
        let mut expected = [
            JsonNode::StartArray,
            JsonNode::StartArray,
            JsonNode::EndArray,
            JsonNode::StartArray,
            JsonNode::StartArray,
            JsonNode::EndArray,
            JsonNode::EndArray,
            JsonNode::EndArray,
        ]
        .into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_array_single_string() {
        let (mut parser, mut input) = setup_parser::<6, 4>(
            "[  \t
         \"hoge\"]",
        );
        let mut expected = [
            JsonNode::StartArray,
            JsonNode::Value(JsonScalarValue::String("hoge")),
            JsonNode::EndArray,
        ]
        .into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
    #[test]
    fn test_parser_array_multiple_values() {
        let (mut parser, mut input) = setup_parser::<8, 4>(
            "[  \t
         \"hoge\",-10.0, true, null, 
         \"  fuga\"]",
        );
        let mut expected = [
            JsonNode::StartArray,
            JsonNode::Value(JsonScalarValue::String("hoge")),
            JsonNode::Value(JsonScalarValue::Number(JsonNumber::F32(-10.0))),
            JsonNode::Value(JsonScalarValue::Boolean(true)),
            JsonNode::Value(JsonScalarValue::Null),
            JsonNode::Value(JsonScalarValue::String("  fuga")),
            JsonNode::EndArray,
        ]
        .into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }

    #[test]
    fn test_parser_map_multiple_values() {
        let (mut parser, mut input) = setup_parser::<8, 4>(
            "{  \t
         \"hoge\":-10.0, true: null, 
         \"  fuga\": \"piyo\"}",
        );
        let mut expected = [
            JsonNode::StartMap,
            JsonNode::Key(JsonScalarValue::String("hoge")),
            JsonNode::Value(JsonScalarValue::Number(JsonNumber::F32(-10.0))),
            JsonNode::Key(JsonScalarValue::Boolean(true)),
            JsonNode::Value(JsonScalarValue::Null),
            JsonNode::Key(JsonScalarValue::String("  fuga")),
            JsonNode::Value(JsonScalarValue::String("piyo")),
            JsonNode::EndMap,
        ]
        .into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }

    #[test]
    fn test_parser_nested_map_array() {
        let (mut parser, mut input) = setup_parser::<8, 4>(
            "{\"array\":[], \"array2\":[[1, 2,], 3], \"map\": {\"nested\": null}}",
        );
        let mut expected = [
            JsonNode::StartMap,
            JsonNode::Key(JsonScalarValue::String("array")),
            JsonNode::StartArray,
            JsonNode::EndArray,
            JsonNode::Key(JsonScalarValue::String("array2")),
            JsonNode::StartArray,
            JsonNode::StartArray,
            JsonNode::Value(JsonScalarValue::Number(JsonNumber::I32(1))),
            JsonNode::Value(JsonScalarValue::Number(JsonNumber::I32(2))),
            JsonNode::EndArray,
            JsonNode::Value(JsonScalarValue::Number(JsonNumber::I32(3))),
            JsonNode::EndArray,
            JsonNode::Key(JsonScalarValue::String("map")),
            JsonNode::StartMap,
            JsonNode::Key(JsonScalarValue::String("nested")),
            JsonNode::Value(JsonScalarValue::Null),
            JsonNode::EndMap,
            JsonNode::EndMap,
        ]
        .into_iter();
        for _ in 0..10 {
            if parser
                .parse(&mut input, |node| {
                    assert_eq!(Some(node), expected.next());
                    DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
                })
                .expect("Parser must not fail.")
            {
                break;
            }
        }
        assert_eq!(None, expected.next()); // All expected numbers are detected.
    }
}
